import torch
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from torch import nn, Tensor
from omegaconf import DictConfig

from einops import rearrange, repeat
from vit_pytorch.cross_vit import CrossTransformer

from .modules.local_attn import LocalEncoder

from external_submodules.hptr.src.models.modules.mlp import MLP
from external_submodules.hptr.src.models.modules.point_net import PointNet
from external_submodules.hptr.src.models.modules.transformer import TransformerBlock
from external_submodules.hptr.src.models.modules.decoder_ensemble import DecoderEnsemble
from external_submodules.hptr.src.models.modules.multi_modal import MultiModalAnchors


class RedMotion(nn.Module):
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
        decoder: DictConfig,
        tf_cfg: DictConfig,
        intra_class_encoder: DictConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_pred = decoder.n_pred
        self.n_decoders = n_decoders
        self.pl_aggr = pl_aggr
        self.pred_subsampling_rate = kwargs.get("pred_subsampling_rate", 1)
        decoder["mlp_head"]["n_step_future"] = decoder["mlp_head"]["n_step_future"] // self.pred_subsampling_rate
        
        self.hidden_dim = hidden_dim
        self.measure_neural_collapse = kwargs.get("measure_neural_collapse", False)

        self.intra_class_encoder = IntraClassEncoder(
            hidden_dim=hidden_dim,
            agent_attr_dim=agent_attr_dim,
            map_attr_dim=map_attr_dim,
            tl_attr_dim=tl_attr_dim,
            pl_aggr=pl_aggr,
            tf_cfg=tf_cfg,
            use_current_tl=use_current_tl,
            n_step_hist=n_step_hist,
            n_pl_node=n_pl_node,
            measure_neural_collapse=self.measure_neural_collapse,
            **intra_class_encoder,
        )

        decoder["tf_cfg"] = tf_cfg
        decoder["hidden_dim"] = hidden_dim
        self.decoder = DecoderEnsemble(n_decoders, decoder_cfg=decoder)

        model_parameters = filter(lambda p: p.requires_grad, self.intra_class_encoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Encoder parameters: {total_params/1000000:.2f}M")
        model_parameters = filter(lambda p: p.requires_grad, self.decoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Decoder parameters: {total_params/1000000:.2f}M")

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
        inference_repeat_n: int = 1,
        inference_cache_map: bool = False,
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
        for _ in range(inference_repeat_n):
            valid = target_valid if self.pl_aggr else target_valid.any(-1)  # [n_scene, n_target]

            if self.measure_neural_collapse:
                emb, emb_invalid, target_embs = self.intra_class_encoder(
                    target_valid=target_valid,
                    target_attr=target_attr,
                    other_valid=other_valid,
                    other_attr=other_attr,
                    map_valid=map_valid,
                    map_attr=map_attr,
                    tl_valid=tl_valid,
                    tl_attr=tl_attr,
                    valid=valid,
                    target_type=target_type,
                )
            else:
                emb, emb_invalid = self.intra_class_encoder(
                    target_valid=target_valid,
                    target_attr=target_attr,
                    other_valid=other_valid,
                    other_attr=other_attr,
                    map_valid=map_valid,
                    map_attr=map_attr,
                    tl_valid=tl_valid,
                    tl_attr=tl_attr,
                    valid=valid,
                    target_type=target_type,
                )

            conf, pred = self.decoder(valid=valid, target_type=target_type, emb=emb, emb_invalid=emb_invalid)

            # Add interpolation here (to invert subsampling)
            if self.pred_subsampling_rate != 1:
                n_decoder, n_scene, n_target, n_pred, n_step_future, pred_dim = pred.shape
                pred = rearrange(
                    pred,
                    "n_decoder n_scene n_target n_pred n_step_future pred_dim -> (n_decoder n_scene n_target n_pred) pred_dim n_step_future",
                )
                pred = F.interpolate(pred, mode="linear", scale_factor=self.pred_subsampling_rate)
                pred = rearrange(
                    pred,
                    "(n_decoder n_scene n_target n_pred) pred_dim n_step_future -> n_decoder n_scene n_target n_pred n_step_future pred_dim",
                    n_decoder=n_decoder, n_scene=n_scene, n_target=n_target, n_pred=n_pred, pred_dim=pred_dim,
                )

        assert torch.isfinite(conf).all()
        assert torch.isfinite(pred).all()

        if self.measure_neural_collapse:
            return valid, conf, pred, target_embs
        else:
            return valid, conf, pred


class IntraClassEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        pl_aggr: bool,
        n_step_hist: int,
        n_pl_node: int,
        tf_cfg: DictConfig,
        use_current_tl: bool,
        add_learned_pe: bool,
        use_point_net: bool,
        n_layer_mlp: int,
        mlp_cfg: DictConfig,
        n_layer_tf: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.pl_aggr = pl_aggr
        self.use_current_tl = use_current_tl
        self.add_learned_pe = add_learned_pe
        self.use_point_net = use_point_net

        self.fc_tl = MLP([tl_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
        if self.use_point_net:
            assert not self.pl_aggr
            self.point_net_target = PointNet(agent_attr_dim, hidden_dim, n_layer=n_layer_mlp, **mlp_cfg)
            self.point_net_other = PointNet(agent_attr_dim, hidden_dim, n_layer=n_layer_mlp, **mlp_cfg)
            self.point_net_map = PointNet(map_attr_dim, hidden_dim, n_layer=n_layer_mlp, **mlp_cfg)
        else:
            self.fc_target = MLP([agent_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
            self.fc_other = MLP([agent_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
            self.fc_map = MLP([map_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)

        if self.add_learned_pe:
            if self.pl_aggr or self.use_point_net:
                self.pe_target = nn.Parameter(torch.zeros([1, hidden_dim]), requires_grad=True)
                self.pe_other = nn.Parameter(torch.zeros([1, 1, hidden_dim]), requires_grad=True)
                self.pe_map = nn.Parameter(torch.zeros([1, 1, hidden_dim]), requires_grad=True)
            else:
                self.pe_target = nn.Parameter(torch.zeros([1, n_step_hist, hidden_dim]), requires_grad=True)
                self.pe_other = nn.Parameter(torch.zeros([1, 1, n_step_hist, hidden_dim]), requires_grad=True)
                self.pe_map = nn.Parameter(torch.zeros([1, 1, n_pl_node, hidden_dim]), requires_grad=True)
            if self.use_current_tl:
                self.pe_tl = nn.Parameter(torch.zeros([1, 1, 1, hidden_dim]), requires_grad=True)
            else:
                self.pe_tl = nn.Parameter(torch.zeros([1, n_step_hist, 1, hidden_dim]), requires_grad=True)

        if not (self.pl_aggr or self.use_point_net):  # singular token in this case
            self.trajectory_encoder = nn.ModuleList(
                [
                    TransformerBlock(d_model=hidden_dim, d_feedforward=hidden_dim * 4, **tf_cfg)
                    for _ in range(3)
                ]
            )
        
        self.red_decoder = ReductionDecoder(
            hidden_dim=hidden_dim,
            tf_cfg=tf_cfg,
            n_descriptors=100,
            n_layer_tf_all2all=3,
        )

        self.local_encoder = LocalEncoder(
            n_blocks=3, # 6
            dim=hidden_dim,
            attn_window=16,   
        )

        self.cross_fusion_block = CrossTransformer(
            sm_dim=hidden_dim,
            lg_dim=hidden_dim,
            depth=3, # 6 
            heads=8,
            dim_head=hidden_dim // 8,
            dropout=0.1,
        )
        self.local_global_fusion_token = nn.Parameter(torch.randn(hidden_dim))
        self.global_local_fusion_token = nn.Parameter(torch.randn(hidden_dim))

        self.use_current_agent_state = kwargs.get("use_current_agent_state")
        self.forward_local_emb = kwargs.get("forward_local_emb")
        self.forward_red_emb = kwargs.get("forward_red_emb")
        self.forward_tl_emb = kwargs.get("forward_tl_emb")

        self.measure_neural_collapse = kwargs.get("measure_neural_collapse")

        print(f"{self.use_current_agent_state = }")
        print(f"{self.forward_local_emb = }")
        print(f"{self.forward_red_emb = }")
        print(f"{self.forward_tl_emb = }")
        print(f"{self.measure_neural_collapse = }")

    def forward(
        self,
        target_valid: Tensor,
        target_attr: Tensor,
        other_valid: Tensor,
        other_attr: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
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

        Returns:
            emb: [n_batch, n_emb, hidden_dim], n_batch = n_scene * n_target
            emb_invalid: [n_batch, n_emb]
        """
        # ! MLP and polyline subnet
        # [n_batch, n_step_hist/1, n_tl, tl_attr_dim]
        tl_valid = tl_valid.flatten(0, 1)
        tl_emb = self.fc_tl(tl_attr.flatten(0, 1), tl_valid)

        # Only use current state of other agents -> no motion data required besides "own" history
        if self.use_current_agent_state:
            other_attr = other_attr[:, :, :, -1:]
            other_valid = other_valid[:, :, :, -1:]

        if self.use_point_net:
            # [n_batch, n_map, map_attr_dim], [n_batch, n_map]
            map_emb, map_valid = self.point_net_map(map_attr.flatten(0, 1), map_valid.flatten(0, 1))
            # [n_batch, n_other, agent_attr_dim], [n_batch, n_other]
            other_emb, other_valid = self.point_net_other(other_attr.flatten(0, 1), other_valid.flatten(0, 1))
            # [n_scene, n_target, agent_attr_dim]
            target_emb, target_valid = self.point_net_target(target_attr, target_valid)
            target_emb = target_emb.flatten(0, 1)  # [n_batch, agent_attr_dim]
            target_valid = target_valid.flatten(0, 1)  # [n_batch]
        else:
            # [n_batch, n_map, (n_pl_node), map_attr_dim]
            map_valid = map_valid.flatten(0, 1)
            map_emb = self.fc_map(map_attr.flatten(0, 1), map_valid)
            # [n_batch, n_other, (n_step_hist), agent_attr_dim]
            other_valid = other_valid.flatten(0, 1)
            other_emb = self.fc_other(other_attr.flatten(0, 1), other_valid)
            # [n_batch, (n_step_hist), agent_attr_dim]
            target_valid = target_valid.flatten(0, 1)
            target_emb = self.fc_target(target_attr.flatten(0, 1), target_valid)
        
        # ! add learned PE
        if self.add_learned_pe:
            tl_emb = tl_emb + self.pe_tl
            map_emb = map_emb + self.pe_map

            if self.use_current_agent_state:
                other_emb = other_emb + self.pe_other[:, :, -1:] # can also be remove as it is temporal with only one time step.. maybe regularizes
            else:
                other_emb = other_emb + self.pe_other

            target_emb = target_emb + self.pe_target

        # ! flatten tokens
        tl_emb = tl_emb.flatten(1, 2)  # [n_batch, (n_step_hist)*n_tl, :]
        tl_valid = tl_valid.flatten(1, 2)  # [n_batch, (n_step_hist)*n_tl]
        if self.pl_aggr or self.use_point_net:
            target_emb = target_emb.unsqueeze(1)  # [n_batch, 1, :]
            target_valid = target_valid.unsqueeze(1)  # [n_batch, 1]
        else:
            # target_emb: [n_batch, n_step_hist/1, :], target_valid: [n_batch, n_step_hist/1]
            map_emb = map_emb.flatten(1, 2)  # [n_batch, n_map*(n_pl_node), :]
            map_valid = map_valid.flatten(1, 2)  # [n_batch, n_map*(n_pl_node)]
            other_emb = other_emb.flatten(1, 2)  # [n_batch, n_other*(n_step_hist), :]
            other_valid = other_valid.flatten(1, 2)  # [n_batch, n_other*(n_step_hist)]
        
        _target_invalid = ~target_valid

        if self.measure_neural_collapse:
            target_embs = []

        for mod in self.trajectory_encoder:
            target_emb, _ = mod(
                src=target_emb, src_padding_mask=_target_invalid, tgt=target_emb, tgt_padding_mask=_target_invalid
            )
            if self.measure_neural_collapse:
                target_embs.append(target_emb)

        env_emb = torch.cat([map_emb, other_emb], dim=1)
        local_valid = torch.cat([map_valid, other_valid], dim=1)

        local_emb = self.local_encoder(
            x=env_emb,
            mask=local_valid,
        )
        
        red_emb = self.red_decoder(
            emb=local_emb,
            emb_invalid=~local_valid,
            valid=kwargs["valid"],
        )

        red_valid = torch.ones(target_valid.shape[0], 100, device="cuda", dtype=torch.bool) # 100 = num of RED tokens

        n_batch = target_emb.shape[0]
        local_fusion_tokens = repeat(self.local_global_fusion_token, "d -> b 1 d", b=n_batch)
        global_fusion_tokens = repeat(self.global_local_fusion_token, "d -> b 1 d", b=n_batch)
        
        fused_local_emb = torch.cat([local_fusion_tokens, target_emb], dim=1)
        fused_global_emb = torch.cat([global_fusion_tokens, red_emb], dim=1)

        fused_local_emb, fused_global_emb = self.cross_fusion_block(
            fused_local_emb, fused_global_emb
        )
        fused_emb = torch.cat([fused_local_emb, fused_global_emb[:, 0][:, None, :]], dim=1)
        fused_valid = torch.ones(n_batch, fused_emb.shape[1], device="cuda", dtype=torch.bool)

        emb = fused_emb
        emb_valid = fused_valid

        if self.forward_tl_emb:
            emb = torch.cat([emb, tl_emb], dim=1)
            emb_valid = torch.cat([emb_valid, tl_valid], dim=1)
        if self.forward_local_emb:
            emb = torch.cat([emb, local_emb], dim=1)
            emb_valid = torch.cat([emb_valid, local_valid], dim=1)
        if self.forward_red_emb:
            emb = torch.cat([emb, red_emb], dim=1)
            emb_valid = torch.cat([emb_valid, red_valid], dim=1)

        emb_invalid = ~emb_valid

        if self.measure_neural_collapse:
            return emb, emb_invalid, target_embs
        else:
            return emb, emb_invalid


class ReductionDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        tf_cfg: DictConfig,
        n_descriptors: int,
        n_layer_tf_all2all: int,
        red_mode: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layer_tf_all2all = n_layer_tf_all2all
        self.n_descriptors = n_descriptors
        self.red_mode = red_mode

        self.learned_embs = MultiModalAnchors(
            hidden_dim=hidden_dim, 
            emb_dim=hidden_dim, 
            n_pred=self.n_descriptors,
            mode_emb="none",
            mode_init="xavier",
            scale=5.0,
            use_agent_type=False, 
        )

        self.decoder = TransformerBlock(
            d_model=hidden_dim,
            d_feedforward=hidden_dim * 4,
            n_layer=n_layer_tf_all2all,
            decoder_self_attn=True,
            **tf_cfg,
        )

    def forward(self, valid: Tensor, emb: Tensor, emb_invalid: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            valid: [n_scene, n_target]
            emb_invalid: [n_scene*n_target, :]
            emb: [n_scene*n_target, :, hidden_dim]

        Returns:
            reduced_emb: [n_scene, n_target, n_descriptors]
        """
        reduced_emb = self.learned_embs(valid.flatten(0, 1), None, None)
        reduced_emb, _ = self.decoder(src=reduced_emb, tgt=emb, tgt_padding_mask=emb_invalid)
        
        return reduced_emb