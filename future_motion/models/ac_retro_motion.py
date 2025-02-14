import time
import torch
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from einops import rearrange
from torch import nn, Tensor
from torch_dct import LinearDCT
from omegaconf import DictConfig
from einops import rearrange, repeat, reduce

from hptr_modules.models.modules.transformer import TransformerBlock
from hptr_modules.models.modules.multi_modal import MultiModalAnchors
from hptr_modules.models.modules.decoder_ensemble import DecoderEnsemble, MLPHead
from hptr_modules.models.modules.mlp import MLP

from future_motion.models.ac_wayformer import InputProjections
from future_motion.models.ac_red_motion import ReductionDecoder


class RetroMotion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        n_pl_node: int,
        use_current_tl: bool,
        pl_aggr: bool,
        n_step_hist: int,
        n_decoders: int,
        joint: bool,
        tf_cfg: DictConfig,
        local_encoder: DictConfig,
        reduction_decoder: DictConfig,
        latent_context_module: DictConfig,
        motion_decoder: DictConfig,
        add_context_pos_enc: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_pred = motion_decoder.n_pred
        self.n_decoders = n_decoders
        self.pl_aggr = pl_aggr
        self.pred_subsampling_rate = kwargs.get("pred_subsampling_rate", 1)
        motion_decoder["mlp_head"]["n_step_future"] = motion_decoder["mlp_head"]["n_step_future"] // self.pred_subsampling_rate
        
        self.add_context_pos_enc = add_context_pos_enc
        
        print(f"{add_context_pos_enc = }")
        
        if add_context_pos_enc:
            n_targets = 8 # TODO: read from global config
            seq_len = (reduction_decoder["n_descriptors"] + 2) * n_targets # + 2 for pos & rot embs 
            self.context_pos_enc = get_pos_encoding(
                seq_len=seq_len, dim=hidden_dim,
            )
            print(f"{self.context_pos_enc.shape = }")
        

        self.local_encoder = InputProjections(
            hidden_dim=hidden_dim,
            agent_attr_dim=agent_attr_dim,
            map_attr_dim=map_attr_dim,
            tl_attr_dim=tl_attr_dim,
            pl_aggr=pl_aggr,
            use_current_tl=use_current_tl,
            n_step_hist=n_step_hist,
            n_pl_node=n_pl_node,
            **local_encoder
        )
        # Opt. include in local encoder
        self.to_ref_pos_emb = nn.Linear(2, hidden_dim)
        self.to_ref_rot_emb = nn.Linear(4, hidden_dim)

        self.reduction_decoder = ReductionDecoder(
            hidden_dim=hidden_dim,
            tf_cfg=tf_cfg,
            **reduction_decoder
        )

        self.latent_context_module = TransformerBlock(
            d_model=hidden_dim, d_feedforward=hidden_dim * 4, **latent_context_module, **tf_cfg
        )

        motion_decoder["tf_cfg"] = tf_cfg
        motion_decoder["hidden_dim"] = hidden_dim
        # self.motion_decoder = DecoderEnsemble(n_decoders, decoder_cfg=motion_decoder)
        
        self.motion_decoder = DualDecoder(**motion_decoder)

        model_parameters = filter(lambda p: p.requires_grad, self.local_encoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Local encoder parameters: {total_params/1000000:.2f}M")
        model_parameters = filter(lambda p: p.requires_grad, self.reduction_decoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Reduction decoder parameters: {total_params/1000000:.2f}M")
        model_parameters = filter(lambda p: p.requires_grad, self.latent_context_module.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Latent context module parameters: {total_params/1000000:.2f}M")
        model_parameters = filter(lambda p: p.requires_grad, self.motion_decoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Motion decoder parameters: {total_params/1000000:.2f}M")

    def forward(
        self,
        target_valid: Tensor,
        target_type: Tensor,
        target_attr: Tensor,
        other_valid: Tensor,
        other_attr: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        freeze_enc_and_dec_0: bool = False,
        pairwise_joint: bool = False,
        additive_decoding: bool = False,
        pred_1_global: bool = False,
        pred_1_skip_context: bool = False,
        edit_pred_0: bool = False,
        agent_0_as_global_ref: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
           target_type: [n_scene, n_target, 3]
            # target history, other history, map
                if pl_aggr:
                    target_valid: [n_scene, n_target], bool
                    target_attr: [n_scene, n_target, agent_attr_dim]
                    other_valid: [n_scene, n_target, n_other], bool
                    other_attr: [n_scene, n_target, n_other, agent_attr_dim]
                    map_valid: [n_scene, n_target, n_map], bool
                    map_attr: [n_scene, n_target, n_map, map_attr_dim]
                else:
                    target_valid: [n_scene, n_target, n_step_hist], bool
                    target_attr: [n_scene, n_target, n_step_hist, agent_attr_dim]
                    other_valid: [n_scene, n_target, n_other, n_step_hist], bool
                    other_attr: [n_scene, n_target, n_other, n_step_hist, agent_attr_dim]
                    map_valid: [n_scene, n_target, n_map, n_pl_node], bool
                    map_attr: [n_scene, n_target, n_map, n_pl_node, map_attr_dim]
            # traffic lights: cannot be aggregated, detections are not tracked.
                if use_current_tl:
                    tl_valid: [n_scene, n_target, 1, n_tl], bool
                    tl_attr: [n_scene, n_target, 1, n_tl, tl_attr_dim]
                else:
                    tl_valid: [n_scene, n_target, n_step_hist, n_tl], bool
                    tl_attr: [n_scene, n_target, n_step_hist, n_tl, tl_attr_dim]

        Returns: will be compared to "output/gt_pos": [n_scene, n_agent, n_step_future, 2]
            valid: [n_scene, n_target]
            conf: [n_decoder, n_scene, n_target, n_pred], not normalized!
            pred: [n_decoder, n_scene, n_target, n_pred, n_step_future, pred_dim]
        """
        valid = target_valid if self.pl_aggr else target_valid.any(-1)  # [n_scene, n_target]
        
        if freeze_enc_and_dec_0:
            with torch.no_grad():
                # print("frozen enc and dec 0")
                target_emb, target_valid, other_emb, other_valid, tl_emb, tl_valid, map_emb, map_valid = self.local_encoder(
                    target_valid=target_valid,
                    target_attr=target_attr,
                    other_valid=other_valid,
                    other_attr=other_attr,
                    map_valid=map_valid,
                    map_attr=map_attr,
                    tl_valid=tl_valid,
                    tl_attr=tl_attr,
                )

                emb = torch.cat([target_emb, other_emb, tl_emb, map_emb], dim=1)
                emb_invalid = ~torch.cat([target_valid, other_valid, tl_valid, map_valid], dim=1)
                
                red_emb = self.reduction_decoder(emb=emb, emb_invalid=emb_invalid, valid=valid)

                ref_pos_emb = self.to_ref_pos_emb(kwargs["ref_pos"].flatten(0, 1).flatten(1, 2))
                ref_rot_emb = self.to_ref_rot_emb(kwargs["ref_rot"].flatten(0, 1).flatten(1, 2))
                
                # Concat & rearrange to learn scene-wide context: n_batch = n_scene (not n_scene * n_agent)
                red_emb = torch.cat([red_emb, ref_pos_emb[:, None, :], ref_rot_emb[:, None, :]], dim=1)
                n_scene, n_agent = valid.shape
                scene_emb = rearrange(red_emb, "(n_scene n_agent) n_token ... -> n_scene (n_agent n_token) ...", n_scene=n_scene, n_agent=n_agent, n_token=red_emb.shape[1])
                scene_emb_invalid = torch.zeros(scene_emb.shape[0], scene_emb.shape[1], device=scene_emb.device, dtype=torch.bool)
                
                # opt rename to global pos enc or scene-wide
                if self.add_context_pos_enc:
                    n_batch = scene_emb.shape[0]
                    context_pos_enc = self.context_pos_enc.to(device=scene_emb.device)
                    context_pos_enc = repeat(context_pos_enc, "seq_len dim -> n_batch seq_len dim", n_batch=n_batch)
                
                scene_emb = scene_emb + context_pos_enc

                scene_emb, _ = self.latent_context_module(src=scene_emb, tgt=scene_emb, tgt_padding_mask=scene_emb_invalid)
        else:
            target_emb, target_valid, other_emb, other_valid, tl_emb, tl_valid, map_emb, map_valid = self.local_encoder(
                target_valid=target_valid,
                target_attr=target_attr,
                other_valid=other_valid,
                other_attr=other_attr,
                map_valid=map_valid,
                map_attr=map_attr,
                tl_valid=tl_valid,
                tl_attr=tl_attr,
            )

            emb = torch.cat([target_emb, other_emb, tl_emb, map_emb], dim=1)
            emb_invalid = ~torch.cat([target_valid, other_valid, tl_valid, map_valid], dim=1)
            
            red_emb = self.reduction_decoder(emb=emb, emb_invalid=emb_invalid, valid=valid)

            ref_pos_emb = self.to_ref_pos_emb(kwargs["ref_pos"].flatten(0, 1).flatten(1, 2))
            ref_rot_emb = self.to_ref_rot_emb(kwargs["ref_rot"].flatten(0, 1).flatten(1, 2))
            
            # Concat & rearrange to learn scene-wide context: n_batch = n_scene (not n_scene * n_agent)
            red_emb = torch.cat([red_emb, ref_pos_emb[:, None, :], ref_rot_emb[:, None, :]], dim=1)
            n_scene, n_agent = valid.shape
            scene_emb = rearrange(red_emb, "(n_scene n_agent) n_token ... -> n_scene (n_agent n_token) ...", n_scene=n_scene, n_agent=n_agent, n_token=red_emb.shape[1])
            scene_emb_invalid = torch.zeros(scene_emb.shape[0], scene_emb.shape[1], device=scene_emb.device, dtype=torch.bool)
            
            if self.add_context_pos_enc:
                n_batch = scene_emb.shape[0]
                context_pos_enc = self.context_pos_enc.to(device=scene_emb.device)
                context_pos_enc = repeat(context_pos_enc, "seq_len dim -> n_batch seq_len dim", n_batch=n_batch)
                
                scene_emb = scene_emb + context_pos_enc

            scene_emb, _ = self.latent_context_module(src=scene_emb, tgt=scene_emb, tgt_padding_mask=scene_emb_invalid)

        # Rearrange again for motion decoding
        emb = rearrange(scene_emb, "n_scene (n_agent n_token) ... -> (n_scene n_agent) n_token ...", n_scene=n_scene, n_agent=n_agent, n_token=red_emb.shape[1])
        emb_invalid = rearrange(scene_emb_invalid, "n_scene (n_agent n_token) -> (n_scene n_agent) n_token", n_scene=n_scene, n_agent=n_agent)
        
        conf_0, pred_0, valid_1, conf_1, pred_1, motion_tokens, retokenized_motion, to_predict_1 = self.motion_decoder(
            valid=valid, target_type=target_type, emb=emb, emb_invalid=emb_invalid,
            ref_pos=kwargs["ref_pos"], ref_rot=kwargs["ref_rot"], ref_role=kwargs["ref_role"],
            freeze_decoder_0=freeze_enc_and_dec_0,
            pairwise_joint=pairwise_joint,
            additive_decoding=additive_decoding,
            pred_1_global=pred_1_global,                                                                                       
            pred_1_skip_context=pred_1_skip_context,
            edit_pred_0=edit_pred_0,
            agent_0_as_global_ref=agent_0_as_global_ref,
        )

        if self.pred_subsampling_rate != 1:
            n_scene, n_target, n_pred, _, pred_dim = pred.shape
            pred = rearrange(
                pred,
                "n_scene n_target n_pred n_step_future pred_dim -> (n_scene n_target n_pred) pred_dim n_step_future",
            )
            pred = F.interpolate(pred, mode="linear", scale_factor=self.pred_subsampling_rate)
            pred = rearrange(
                pred,
                "(n_scene n_target n_pred) pred_dim n_step_future -> n_scene n_target n_pred n_step_future pred_dim",
                n_decoder=n_decoder, n_scene=n_scene, n_target=n_target, n_pred=n_pred, pred_dim=pred_dim,
            )

        assert torch.isfinite(conf_0).all()
        assert torch.isfinite(pred_0).all()
        assert torch.isfinite(conf_1).all()
        assert torch.isfinite(pred_1).all()
        
        return motion_tokens, retokenized_motion, valid, conf_0[None, ...], pred_0[None, ...], valid_1, conf_1[None, ...], pred_1[None, ...], to_predict_1 # Add n_decoder dim
    
    
class DualDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_pred: int,
        use_vmap: bool,
        mlp_head: DictConfig,
        multi_modal_anchors: DictConfig,
        tf_cfg: DictConfig,
        n_decoder_layers: int,
        anchor_self_attn: bool,
        n_dct_coeffs: int = 0,
        dct_only_for_pos: bool = False,
        use_fan_layer: bool = False,
        retokenizer_without_dct: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_pred = n_pred
        self.n_dct_coeffs = n_dct_coeffs
        self.dct_only_for_pos = dct_only_for_pos
        self.retokenizer_without_dct = retokenizer_without_dct # TODO: dbl check tokenizer/ embedding layer naming etc. not tokenization as in MotionLM nor as in LLMs
        
        print(f"{use_fan_layer = }")
        print(f"{dct_only_for_pos = }")
        print(f"{retokenizer_without_dct = }")

        
        self.motion_anchors = MultiModalAnchors(
            hidden_dim=hidden_dim, emb_dim=hidden_dim, n_pred=n_pred, **multi_modal_anchors
        )
        self.transformer_blocks = nn.ModuleList((
            TransformerBlock(
                d_model=hidden_dim,
                d_feedforward=hidden_dim * 4,
                n_layer=n_decoder_layers // 2,
                decoder_self_attn=anchor_self_attn,
                **tf_cfg,
            ),
            TransformerBlock(
                d_model=hidden_dim,
                d_feedforward=hidden_dim * 4,
                n_layer=n_decoder_layers // 2,
                decoder_self_attn=anchor_self_attn,
                **tf_cfg,
            ),
            TransformerBlock(
                d_model=hidden_dim,
                d_feedforward=hidden_dim * 4,
                n_layer=n_decoder_layers // 2,
                decoder_self_attn=anchor_self_attn,
                **tf_cfg,
            ),
        ))
        
        if n_dct_coeffs:
            self.linear_dct = LinearDCT(in_features=mlp_head["n_step_future"], type="dct", norm="ortho")
            self.linear_idct = LinearDCT(in_features=mlp_head["n_step_future"], type="idct", norm="ortho")
            self.n_step_future = mlp_head["n_step_future"]
            
            if dct_only_for_pos: # Assume some cov params (better call density params as no cov for Laplace distributions), designed to forecast motion as probability densities, opt. add assert to ensure that
                n_density_params = int(mlp_head["predictions"][-1][-1]) * mlp_head["n_step_future"]
                mlp_head["n_step_future"] = n_dct_coeffs * 2 + n_density_params # * 2 for x and y param per pos
                print(f"{n_density_params = }")
            else:    
                mlp_head["n_step_future"] = n_dct_coeffs # Predict DCT coeffs instead of all waypoints
            
            print(f"{n_dct_coeffs = }")
        
        # unembed 
        self.detokenizers = nn.ModuleList((
            MLPHead(hidden_dim=hidden_dim, use_vmap=use_vmap, n_pred=n_pred, dct_only_for_pos=dct_only_for_pos, **mlp_head),
            MLPHead(hidden_dim=hidden_dim, use_vmap=use_vmap, n_pred=n_pred, dct_only_for_pos=dct_only_for_pos, **mlp_head),
        ))
        
        # reembed (the regression output)
        # not for the initial anchor, but a later hidden state -> re... opt. call Retrotokenizer (retrocausation)
        _d = 2 * hidden_dim # opt. also optimize as hyperparam
        n_timesteps = 80
        pred_dim = 5 # assuming "cov3" TODO: read from config
        
        if retokenizer_without_dct:
            self.retokenizer = MLP(
                fc_dims=[n_timesteps * pred_dim, _d, _d, hidden_dim],
            )
        else:
            self.retokenizer = MLP(
                fc_dims=[self.detokenizers[0].mlp_pred._decoders[0].output_dim, _d, _d, hidden_dim],
            )

    # rename emb to context emb, and anchors to anchor_emb
    def forward(
        self, valid: Tensor, target_type: Tensor, emb: Tensor, emb_invalid: Tensor, ref_rot, ref_pos, ref_role,
        freeze_decoder_0: bool = False,
        pairwise_joint: bool = False,
        edit_pred_0: bool = False,
        additive_decoding: bool = False,
        pred_1_global: bool = False,
        pred_1_skip_context: bool = False,
        agent_0_as_global_ref: bool = False,
        return_last_hidden_state: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            valid: [n_scene, n_target]
            emb_invalid: [n_scene * n_target, :]
            emb: [n_scene * n_target, :, hidden_dim]
            target_type: [n_scene, n_target, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]

        Returns:
            conf: [n_scene, n_target, n_pred]
            pred: [n_scene, n_target, n_pred, n_step_future, pred_dim]
        """
        # Freeze anchors and detokenizers/ unembed as well?
        if freeze_decoder_0:
            with torch.no_grad():
                # print("frozen dec 0")
                # [n_batch, n_pred, hidden_dim]
                motion_anchors = self.motion_anchors(valid.flatten(0, 1), emb, target_type.flatten(0, 1))
                
                # [n_batch, n_pred, hidden_dim]
                motion_tokens, _ = self.transformer_blocks[0](src=motion_anchors, tgt=emb, tgt_padding_mask=emb_invalid)
                
                # [n_scene, n_target, n_pred, hidden_dim]
                motion_emb_0 = motion_tokens.view(valid.shape[0], valid.shape[1], self.n_pred, self.hidden_dim)
                conf_0, pred_0 = self.detokenizers[0](valid, motion_emb_0, target_type)
        else:
            # [n_batch, n_pred, hidden_dim]
            motion_anchors = self.motion_anchors(valid.flatten(0, 1), emb, target_type.flatten(0, 1))
            
            # [n_batch, n_pred, hidden_dim]
            motion_tokens, _ = self.transformer_blocks[0](src=motion_anchors, tgt=emb, tgt_padding_mask=emb_invalid)
            
            # [n_scene, n_target, n_pred, hidden_dim]
            motion_emb_0 = motion_tokens.view(valid.shape[0], valid.shape[1], self.n_pred, self.hidden_dim)
            
            if return_last_hidden_state:
                conf_0, pred_0, last_hidden_state_0 = self.detokenizers[0](valid, motion_emb_0, target_type, return_last_hidden_state=True)
            else:
                conf_0, pred_0 = self.detokenizers[0](valid, motion_emb_0, target_type)
        
        
        if self.n_dct_coeffs:
            if self.dct_only_for_pos:
                pred_0 = rearrange(pred_0, "... n_params pred_dim -> ... pred_dim n_params") # n_params = 2 * n_dct_coeffs + n_density_params
                n_scene, n_target, n_pred, pred_dim, _ = pred_0.shape
                pred_0_dct = torch.zeros((n_scene, n_target, n_pred, pred_dim, self.n_step_future * 2), device=pred_0.device)
                pred_0_dct[..., :self.n_dct_coeffs * 2] = pred_0[..., :self.n_dct_coeffs * 2]
                
                pred_0_dct = rearrange(pred_0_dct, "... 1 (n_step_future xy) ->  ... xy n_step_future", xy=2)

                
                # Inverse discrete cosine transform: n_dct_coeffs -> n_step_future
                pred_0_pos = self.linear_idct(pred_0_dct)
                
                pred_0_density = pred_0[..., self.n_dct_coeffs * 2:]
                pred_0_density = rearrange(pred_0_density, "... 1 (n_step_future params) -> ... params n_step_future", n_step_future=self.n_step_future)
                
                
                pred_0 = torch.cat((pred_0_pos, pred_0_density), dim=-2)
                pred_0 = rearrange(pred_0, "... pred_dim n_step_future -> ... n_step_future pred_dim")
            else:
                pred_0 = rearrange(pred_0, "... n_dct_coeffs pred_dim -> ... pred_dim n_dct_coeffs")
                n_scene, n_target, n_pred, pred_dim, _ = pred_0.shape
                pred_0_dct = torch.zeros((n_scene, n_target, n_pred, pred_dim, self.n_step_future), device=pred_0.device)
                pred_0_dct[..., :self.n_dct_coeffs] = pred_0
                
                # Inverse discrete cosine transform: n_dct_coeffs -> n_step_future
                pred_0 = self.linear_idct(pred_0_dct)
                pred_0 = rearrange(pred_0, "... pred_dim n_step_future -> ... n_step_future pred_dim")
            
        
        pred_0_pos = pred_0[..., :2]
        
        # Edit trajs here
        if edit_pred_0:
            # pred_0_pos[..., 79, :] += 20
            pred_0_pos[..., 70:80, :] += 20
            # print(f"{pred_0_pos.shape = }") # [16, 8, 6, 80, 2]
            
            # edit last point 
            
            # take half of the points and upsample
            # pred_0_pos = pred_0_pos[..., :40, :]
            # n_scene, n_target, n_pred, _, pred_dim = pred_0_pos.shape
            # pred_0_pos = rearrange(
            #     pred_0_pos,
            #     "n_scene n_target n_pred n_step_future pred_dim -> (n_scene n_target n_pred) pred_dim n_step_future",
            # )
            # pred_0_pos = F.interpolate(pred_0_pos, mode="linear", scale_factor=2)
            # pred_0_pos = rearrange(
            #     pred_0_pos,
            #     "(n_scene n_target n_pred) pred_dim n_step_future -> n_scene n_target n_pred n_step_future pred_dim",
            #     n_scene=n_scene, n_target=n_target, n_pred=n_pred, pred_dim=pred_dim,
            # )
            # print(f"{pred_0_pos.shape = }") # [16, 8, 6, 80, 2]
            # time.sleep(100)
            
            # To "zero out"
            # pred_0_pos = torch.zeros_like(pred_0_pos)
        
        # # debug
        # pred_0_pos = torch.zeros_like(pred_0_pos)
        
        if agent_0_as_global_ref:
            # transform all forecasts to local reference frame of agent 0 
            # 2 steps all to global, then all to agent 0's ref frame, later opt simplify and combine to one operation by multiplying the matrices? prob. less intuitive
            pred_0_global = torch_pos2global(pred_0_pos.flatten(2, 3), ref_pos, ref_rot)
            n_agent = ref_pos.shape[1]
            
            # row-major order in pytorch
            ref_pos_agent_0 = repeat(ref_pos[:, 0:1], "n_scene 1 row col -> n_scene n_agent row col", n_agent=n_agent)
            ref_rot_agent_0 = repeat(ref_rot[:, 0:1], "n_scene 1 row col -> n_scene n_agent row col", n_agent=n_agent)
            
            # global in view of agent 0
            pred_0_global = torch_pos2local(pred_0_global, ref_pos_agent_0, ref_rot_agent_0)
        else: 
            pred_0_global = torch_pos2global(pred_0_pos.flatten(2, 3), ref_pos, ref_rot)
        
        pred_0_global = rearrange(pred_0_global, "n_scene n_agent (n_pred n_step_future) xy -> n_scene n_agent n_pred n_step_future xy", n_pred=self.n_pred)
        
        # concat Gaussian covs
        pred_0_global = torch.concat((pred_0_global, pred_0[..., 2:]), dim=-1)
        
        if self.n_dct_coeffs and not self.retokenizer_without_dct:
            if self.dct_only_for_pos:
                pred_0_global_pos = pred_0_global[..., :2]
                pred_0_global_pos = rearrange(pred_0_global_pos, "... n_step_future pred_dim -> ... pred_dim n_step_future")
                pred_0_global_pos = self.linear_dct(pred_0_global_pos)[..., :self.n_dct_coeffs]
                pred_0_global_pos = rearrange(pred_0_global_pos, "... xy n_dct_coeffs -> ... (n_dct_coeffs xy) 1")
                
                # pred_dim = 1 since both dct coeffs for pos and density params are stored in n_step_future, yet with different frequencies
                pred_0_global_density = rearrange(pred_0[..., 2:], "... n_step_future dim -> ... (n_step_future dim) 1")
                
                pred_0_global = torch.cat((pred_0_global_pos, pred_0_global_density), dim=-2)
            else:
                pred_0_global = rearrange(pred_0_global, "... n_step_future pred_dim -> ... pred_dim n_step_future")
                pred_0_global = self.linear_dct(pred_0_global)[..., :self.n_dct_coeffs]
                pred_0_global = rearrange(pred_0_global, "... pred_dim n_dct_coeffs -> ... n_dct_coeffs pred_dim")
                
        
        # Also concat pred conf? -> no not meaningful in a joint sense, therefore let the model "relearn"/ regenerate that
        pred_0_tokenlike = rearrange(pred_0_global, "n_scene n_target n_pred n_timesteps pred_dim -> (n_scene n_target) n_pred (n_timesteps pred_dim)")
        valid_tokenlike = rearrange(valid, "n_scene n_agent -> (n_scene n_agent)")
        valid_tokenlike = repeat(valid_tokenlike, "... -> ... n_pred", n_pred=self.n_pred)
        
        retokenized_motion = self.retokenizer(pred_0_tokenlike, valid_tokenlike)
        
        # global attn across all agents and modes (n_traget, n_pred)
        n_scene, n_target, *_ = pred_0.shape
        
        to_predict = ref_role[..., 2]
        valid_only_interactive = False
        
        if pairwise_joint:
            # filter out non-predict agents
            
            # opt add random selection here during training because the train samples contain more than 2 targets with role = to predict
            # ideally implement with dynamic check that triggers the rand selection if more than 2 targets have role = to predict
            # opt also add another reduced_joint mode with a configurable number of selected target agents
            # see below first two are even better than random as these are often interacting targets
            
            if to_predict.sum(dim=-1).min() < 2.0 or to_predict.sum(dim=-1).max() > 2.0: # .mean() doesn't work for boolean tensors...
                to_predict = torch.zeros(to_predict.shape, dtype=torch.bool, device=to_predict.device)
                to_predict[:, 0:2] = True # fixed batch size
                
                # print(f"{ref_role[..., 1]}") # targets with interactive behavior are always at index 0, 1 in train set
                # https://waymo.com/open/data/motion/tfexample object of interest = with interactive behavior

            motion_tokens_global = rearrange(retokenized_motion, "(n_scene n_target) n_pred hidden_dim -> n_scene n_target n_pred hidden_dim", n_scene=n_scene)
            
            
            motion_tokens_global = motion_tokens_global[to_predict]
            motion_tokens_global = rearrange(motion_tokens_global, "(n_scene n_role_to_predict) n_pred hidden_dim -> n_scene (n_role_to_predict n_pred) hidden_dim", n_scene=n_scene)
            
            valid_tokenlike = rearrange(valid_tokenlike, "(n_scene n_target) n_pred -> n_scene n_target n_pred", n_scene=n_scene)
            valid_tokenlike = valid_tokenlike[to_predict]
            
            emb_invalid = rearrange(emb_invalid, "(n_scene n_target) ... -> n_scene n_target ...", n_scene=n_scene)
            emb = rearrange(emb, "(n_scene n_target) ... -> n_scene n_target ...", n_scene=n_scene)
            emb_invalid = emb_invalid[to_predict]
            emb = emb[to_predict]
            
            valid = valid[to_predict]
            valid = rearrange(valid, "(n_scene n_role_to_predict) -> n_scene n_role_to_predict", n_scene=n_scene)
        else:
            motion_tokens_global = rearrange(retokenized_motion, "(n_scene n_target) n_pred hidden_dim -> n_scene (n_target n_pred) hidden_dim", n_scene=n_scene)
        
        valid_tokenlike_global = rearrange(valid_tokenlike, "(n_scene n_target) n_pred -> n_scene (n_target n_pred)", n_scene=n_scene)
        motion_tokens_global, _ = self.transformer_blocks[1](src=motion_tokens_global, tgt=motion_tokens_global, tgt_padding_mask=valid_tokenlike_global)
        
        # local alignment with scene context - map traffic light states etc.
         
        motion_tokens_local = rearrange(motion_tokens_global, "n_scene (n_target n_pred) hidden_dim -> (n_scene n_target) n_pred hidden_dim", n_pred=self.n_pred)
        if pred_1_skip_context:
            motion_tokens_1 = motion_tokens_local
        else:
            if agent_0_as_global_ref:
                # TODO: only use scene embedding of agent 0 as context for all agents (used as global yet learned data-effient encoder is reused for other agent-centric views as well (for mraginal preds))
                emb = rearrange(emb, "(n_scene n_joint_agents) ... -> n_scene n_joint_agents ...", n_scene=n_scene)
                emb_invalid = rearrange(emb_invalid, "(n_scene n_joint_agents) ... -> n_scene n_joint_agents ...", n_scene=n_scene)
                n_joint_agents = emb.shape[1]
                
                emb = repeat(emb[:, 0:1], "n_scene 1 ... -> n_scene n_joint_agents ...", n_joint_agents=n_joint_agents)
                emb_invalid = repeat(emb_invalid[:, 0:1], "n_scene 1 ... -> n_scene n_joint_agents ...", n_joint_agents=n_joint_agents)
                
                emb = rearrange(emb, "n_scene n_joint_agents ... -> (n_scene n_joint_agents) ...")
                emb_invalid = rearrange(emb_invalid, "n_scene n_joint_agents ... -> (n_scene n_joint_agents) ...")
                
            motion_tokens_1, _ = self.transformer_blocks[2](src=motion_tokens_local, tgt=emb, tgt_padding_mask=emb_invalid)
        
        motion_emb_1 = motion_tokens_1.view(valid.shape[0], valid.shape[1], self.n_pred, self.hidden_dim)
        conf_1, pred_1 = self.detokenizers[1](valid, motion_emb_1, target_type)
        
        if self.n_dct_coeffs:
            if self.dct_only_for_pos:
                pred_1 = rearrange(pred_1, "... n_params pred_dim -> ... pred_dim n_params") # n_params = 2 * n_dct_coeffs + n_density_params
                n_scene, n_target, n_pred, pred_dim, _ = pred_1.shape
                pred_1_dct = torch.zeros((n_scene, n_target, n_pred, pred_dim, self.n_step_future * 2), device=pred_1.device)
                pred_1_dct[..., :self.n_dct_coeffs * 2] = pred_1[..., :self.n_dct_coeffs * 2]
                
                pred_1_dct = rearrange(pred_1_dct, "... 1 (n_step_future xy) ->  ... xy n_step_future", xy=2)
                
                # Inverse discrete cosine transform: n_dct_coeffs -> n_step_future
                pred_1_pos = self.linear_idct(pred_1_dct)
                
                pred_1_density = pred_1[..., self.n_dct_coeffs * 2:]
                pred_1_density = rearrange(pred_1_density, "... 1 (n_step_future params) -> ... params n_step_future", n_step_future=self.n_step_future)
                
                
                pred_1 = torch.cat((pred_1_pos, pred_1_density), dim=-2)
                pred_1 = rearrange(pred_1, "... pred_dim n_step_future -> ... n_step_future pred_dim")
            else:
                pred_1 = rearrange(pred_1, "... n_dct_coeffs pred_dim -> ... pred_dim n_dct_coeffs")
                n_scene, n_target, n_pred, pred_dim, _ = pred_1.shape
                pred_1_dct = torch.zeros((n_scene, n_target, n_pred, pred_dim, self.n_step_future), device=pred_1.device)
                pred_1_dct[..., :self.n_dct_coeffs] = pred_1
                
                # Inverse discrete cosine transform: n_dct_coeffs -> n_step_future
                pred_1 = self.linear_idct(pred_1_dct)
                pred_1 = rearrange(pred_1, "... pred_dim n_step_future -> ... n_step_future pred_dim")
            
        if agent_0_as_global_ref:
            # back to per-agent local
            pred_1_pos = pred_1[..., :2]
            _pred_1_global = torch_pos2global(pred_1_pos.flatten(2, 3), ref_pos_agent_0[:, :n_joint_agents], ref_rot_agent_0[:, :n_joint_agents])
            
            # to local views of all agents ("regular ones")
            pred_1_local = torch_pos2local(_pred_1_global, ref_pos[:, :n_joint_agents], ref_rot[:, :n_joint_agents])
            
            pred_1_local = rearrange(pred_1_local, "n_scene n_agent (n_pred n_step_future) xy -> n_scene n_agent n_pred n_step_future xy", n_pred=self.n_pred)
            pred_1 = torch.concat((pred_1_local, pred_1[..., 2:]), dim=-1) # pos/loc + scale params
        
        if pred_1_global:
            pred_1_pos = pred_1[..., :2]
            
            pred_1_local = torch_pos2local(pred_1_pos.flatten(2, 3), ref_pos[:, :2], ref_rot[:, :2])
            pred_1_local = rearrange(pred_1_local, "n_scene n_agent (n_pred n_step_future) xy -> n_scene n_agent n_pred n_step_future xy", n_pred=self.n_pred)
            pred_1 = torch.concat((pred_1_local, pred_1[..., 2:]), dim=-1)
    
        if additive_decoding:
            pred_1 = pred_0[..., :2, :, :, :] + pred_1

            return conf_0, pred_0, valid, conf_1, pred_1, motion_tokens, retokenized_motion, to_predict
        
        if return_last_hidden_state:
            return conf_0, pred_0, valid, conf_1, pred_1, motion_tokens, retokenized_motion, to_predict, last_hidden_state_0
            
        
        return conf_0, pred_0, valid, conf_1, pred_1, motion_tokens, retokenized_motion, to_predict
            
    
        
def torch_pos2global(in_pos: Tensor, local_pos: Tensor, local_rot: Tensor) -> Tensor:
    """Reverse torch_pos2local

    Args:
        in_pos: [..., M, 2]
        local_pos: [..., 1, 2] translation global to local reference frame
        local_rot: [..., 2, 2] rotation global to local reference frame

    Returns:
        out_pos: [..., M, 2]
    """
    return torch.matmul(in_pos, local_rot.transpose(-1, -2)) + local_pos


def torch_pos2local(in_pos: Tensor, local_pos: Tensor, local_rot: Tensor) -> Tensor:
    """Transform M position to the local coordinates.

    Args:
        in_pos: [..., M, 2]
        local_pos: [..., 1, 2]
        local_rot: [..., 2, 2]

    Returns:
        out_pos: [..., M, 2]
    """
    return torch.matmul(in_pos - local_pos, local_rot)


def get_pos_encoding(seq_len, dim, scaling_factor=10000):
    enc = torch.zeros((seq_len, dim))

    for k in range(seq_len):
        for i in range(int(dim / 2)):
            denominator = torch.tensor(scaling_factor ** (2 * i / dim))
            enc[k, 2 * i] = torch.sin(k / denominator)
            enc[k, 2 * i + 1] = torch.cos(k / denominator)
    
    return enc