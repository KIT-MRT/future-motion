# From implicit past to explicit future motion

import time
import torch
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from torch import nn, Tensor
from omegaconf import DictConfig
from einops import rearrange, repeat

from vit_pytorch import vivit
from vit_pytorch.extractor import Extractor

from external_submodules.hptr.src.models.modules.mlp import MLP
from external_submodules.hptr.src.models.modules.point_net import PointNet
from external_submodules.hptr.src.models.modules.transformer import TransformerBlock
from external_submodules.hptr.src.models.modules.multi_modal import MultiModalAnchors
from external_submodules.hptr.src.models.modules.decoder_ensemble import DecoderEnsemble, MLPHead

from models.ac_red_motion import ReductionDecoder

# use all frame as input and a larger patch size frame wise to balance compute
# and focus on motion (ego motion and motion of others relative to the ego vehicle)
# opt. only use current frame after lidar encoder to forward less tokens



class ImEx(nn.Module):
    def __init__(
        self,
        n_target: int,
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
        input_projections: DictConfig,
        reduction_decoder: DictConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_target = n_target
        self.n_pred = decoder.n_pred
        self.n_decoders = n_decoders
        self.pl_aggr = pl_aggr
        self.pred_subsampling_rate = kwargs.get("pred_subsampling_rate", 1)
        decoder["mlp_head"]["n_step_future"] = decoder["mlp_head"]["n_step_future"] // self.pred_subsampling_rate

        self.input_projections = InputProjections(
            hidden_dim=hidden_dim,
            agent_attr_dim=agent_attr_dim,
            map_attr_dim=map_attr_dim,
            tl_attr_dim=tl_attr_dim,
            pl_aggr=pl_aggr,
            use_current_tl=use_current_tl,
            n_step_hist=n_step_hist,
            n_pl_node=n_pl_node,
            **input_projections
        )

        lidar_vit = vivit.ViT(
            image_size=1024,
            frames=n_step_hist, # 11
            image_patch_size=32,
            frame_patch_size=1, # Temporal attention is rather cheap compared to spatial with factorized attn 
            spatial_depth=6,
            temporal_depth=6,
            dim=hidden_dim,
            mlp_dim=hidden_dim * 4,
            heads=8,
            num_classes=3, # Dummy value as not used for classification
            pool="mean",
            variant="factorized_self_attention",
        )

        self.lidar_encoder = Extractor(
            lidar_vit, layer_name="factorized_transformer"
        )

        self.reduction_decoder = ReductionDecoder(
            hidden_dim=hidden_dim,
            tf_cfg=tf_cfg,
            **reduction_decoder
        )

        decoder["tf_cfg"] = tf_cfg
        decoder["hidden_dim"] = hidden_dim
        self.decoder = Decoder(**decoder)

        model_parameters = filter(lambda p: p.requires_grad, self.input_projections.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Input projections parameters: {total_params/1000000:.2f}M")
        model_parameters = filter(lambda p: p.requires_grad, self.lidar_encoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"LiDAR encoder parameters: {total_params/1000000:.2f}M")
        # model_parameters = filter(lambda p: p.requires_grad, self.encoder.parameters())
        # total_params = sum([np.prod(p.size()) for p in model_parameters])
        # print(f"Encoder parameters: {total_params/1000000:.2f}M")
        model_parameters = filter(lambda p: p.requires_grad, self.decoder.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Decoder parameters: {total_params/1000000:.2f}M")

    def forward(
        self,
        lidar_pillars: Tensor,
        sdc_valid: Tensor,
        sdc_type: Tensor,
        sdc_attr: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        target_valid: Tensor,
        target_type: Tensor,
        inference_repeat_n: int = 1,
        inference_cache_map: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
           sdc_type: [n_scene, n_target, 3]
            # target history, other history, map
                if pl_aggr:
                    sdc_valid: [n_scene, n_target], bool
                    sdc_attr: [n_scene, n_target, agent_attr_dim]
                    other_valid: [n_scene, n_target, n_other], bool
                    other_attr: [n_scene, n_target, n_other, agent_attr_dim]
                    map_valid: [n_scene, n_target, n_map], bool
                    map_attr: [n_scene, n_target, n_map, map_attr_dim]
                else:
                    sdc_valid: [n_scene, n_target, n_step_hist], bool
                    sdc_attr: [n_scene, n_target, n_step_hist, agent_attr_dim]
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
            target_valid = target_valid if self.pl_aggr else target_valid.any(-1)  # [n_scene, n_target]
            valid = sdc_valid if self.pl_aggr else sdc_valid.any(-1)  # [n_scene, 1]
            sdc_emb, sdc_valid, tl_emb, tl_valid, map_emb, map_valid = self.input_projections(
                sdc_valid=sdc_valid,
                sdc_attr=sdc_attr,
                map_valid=map_valid,
                map_attr=map_attr,
                tl_valid=tl_valid,
                tl_attr=tl_attr,
            )

            _, lidar_emb = self.lidar_encoder(rearrange(lidar_pillars, "b f c h w -> b c f h w"))
            lidar_emb = rearrange(lidar_emb, "n_scene n_step_hist n_token hidden_dim -> n_scene (n_step_hist n_token) hidden_dim")
            lidar_valid = torch.ones(lidar_emb.shape[:2], dtype=torch.bool, device=lidar_emb.device)

            emb = torch.cat((lidar_emb, sdc_emb, map_emb, tl_emb), dim=1)
            emb_valid = torch.cat((lidar_valid, sdc_valid, map_valid, tl_valid), dim=1)

            reduced_emb = self.reduction_decoder(emb=emb, emb_invalid=~emb_valid, valid=valid)
            reduced_emb_invalid = torch.zeros(reduced_emb.shape[:2], dtype=torch.bool, device=reduced_emb.device)

            # [n_decoder, n_scene, n_det, n_pred], [n_decoder, n_scene, n_det, n_pred, n_step_future / subsampling_rate, pred_dim]
            pos_rot_pred, cls_pred, traj_conf, traj_pred = self.decoder(n_scene=valid.shape[0], target_type=target_type, emb=reduced_emb, emb_invalid=reduced_emb_invalid)


            if self.pred_subsampling_rate != 1:
                n_decoder, n_scene, n_target, n_pred, n_step_future, pred_dim = traj_pred.shape
                traj_pred = rearrange(
                    traj_pred,
                    "n_decoder n_scene n_target n_pred n_step_future pred_dim -> (n_decoder n_scene n_target n_pred) pred_dim n_step_future",
                )
                traj_pred = F.interpolate(traj_pred, mode="linear", scale_factor=self.pred_subsampling_rate)
                traj_pred = rearrange(
                    traj_pred,
                    "(n_decoder n_scene n_target n_pred) pred_dim n_step_future -> n_decoder n_scene n_target n_pred n_step_future pred_dim",
                    n_decoder=n_decoder, n_scene=n_scene, n_target=n_target, n_pred=n_pred, pred_dim=pred_dim,
                )

        assert torch.isfinite(traj_conf).all()
        assert torch.isfinite(traj_pred).all()
        return pos_rot_pred, cls_pred, target_valid, traj_conf, traj_pred


class InputProjections(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        pl_aggr: bool,
        n_step_hist: int,
        n_pl_node: int,
        use_current_tl: bool,
        add_learned_pe: bool,
        use_point_net: bool,
        n_layer_mlp: int,
        mlp_cfg: DictConfig,
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
            self.point_net_map = PointNet(map_attr_dim, hidden_dim, n_layer=n_layer_mlp, **mlp_cfg)
        else:
            self.fc_target = MLP([agent_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
            self.fc_map = MLP([map_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)

        if self.add_learned_pe:
            if self.pl_aggr or self.use_point_net:
                self.pe_target = nn.Parameter(torch.zeros([1, hidden_dim]), requires_grad=True)
                self.pe_map = nn.Parameter(torch.zeros([1, 1, hidden_dim]), requires_grad=True)
            else:
                self.pe_target = nn.Parameter(torch.zeros([1, n_step_hist, hidden_dim]), requires_grad=True)
                self.pe_map = nn.Parameter(torch.zeros([1, 1, n_pl_node, hidden_dim]), requires_grad=True)
            if self.use_current_tl:
                self.pe_tl = nn.Parameter(torch.zeros([1, 1, 1, hidden_dim]), requires_grad=True)
            else:
                self.pe_tl = nn.Parameter(torch.zeros([1, n_step_hist, 1, hidden_dim]), requires_grad=True)

    def forward(
        self,
        sdc_valid: Tensor,
        sdc_attr: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
    ) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:
        """
        Args:
            # target history, other history, map
                if pl_aggr:
                    sdc_valid: [n_scene, n_target], bool
                    sdc_attr: [n_scene, n_target, agent_attr_dim]
                    map_valid: [n_scene, n_target, n_map], bool
                    map_attr: [n_scene, n_target, n_map, map_attr_dim]
                else:
                    sdc_valid: [n_scene, n_target, n_step_hist], bool
                    sdc_attr: [n_scene, n_target, n_step_hist, agent_attr_dim]
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
            sdc_emb: [n_batch, 1 or n_step_hist, hidden_dim], n_batch = n_scene * n_target (agent-centric)
            sdc_valid: [n_batch, 1 or n_step_hist]
            tl_emb: [n_batch, n_tl * n_step_hist, hidden_dim]
            tl_valid: [n_batch, n_tl * n_step_hist]
            map_emb: [n_batch, n_map or n_map * n_pl_node, hidden_dim]
            map_valid: [n_batch, n_map or n_map * n_pl_node]
        """
        # [n_batch, n_step_hist/1, n_tl, tl_attr_dim]
        tl_valid = tl_valid.flatten(0, 1)
        tl_emb = self.fc_tl(tl_attr.flatten(0, 1), tl_valid)

        if self.use_point_net:
            # [n_batch, n_map, map_attr_dim], [n_batch, n_map]
            map_emb, map_valid = self.point_net_map(map_attr.flatten(0, 1), map_valid.flatten(0, 1))
            # [n_scene, n_target, agent_attr_dim]
            sdc_emb, sdc_valid = self.point_net_target(sdc_attr, sdc_valid)
            sdc_emb = sdc_emb.flatten(0, 1)  # [n_batch, agent_attr_dim]
            sdc_valid = sdc_valid.flatten(0, 1)  # [n_batch]
        else:
            # [n_batch, n_map, (n_pl_node), map_attr_dim]
            map_valid = map_valid.flatten(0, 1)
            map_emb = self.fc_map(map_attr.flatten(0, 1), map_valid)
            # [n_batch, (n_step_hist), agent_attr_dim]
            sdc_valid = sdc_valid.flatten(0, 1)
            sdc_emb = self.fc_target(sdc_attr.flatten(0, 1), sdc_valid)

        if self.add_learned_pe:
            tl_emb = tl_emb + self.pe_tl
            map_emb = map_emb + self.pe_map
            sdc_emb = sdc_emb + self.pe_target

        tl_emb = tl_emb.flatten(1, 2)  # [n_batch, n_step_hist * n_tl, :]
        tl_valid = tl_valid.flatten(1, 2)  # [n_batch, n_step_hist * n_tl]
        if self.pl_aggr or self.use_point_net:
            sdc_emb = sdc_emb.unsqueeze(1)  # [n_batch, 1, :]
            sdc_valid = sdc_valid.unsqueeze(1)  # [n_batch, 1]
        else:
            # sdc_emb: [n_batch, n_step_hist/1, :], sdc_valid: [n_batch, n_step_hist/1]
            map_emb = map_emb.flatten(1, 2)  # [n_batch, n_map * n_pl_node, :]
            map_valid = map_valid.flatten(1, 2)  # [n_batch, n_map * n_pl_node]

        return (
            sdc_emb, sdc_valid,
            tl_emb, tl_valid,
            map_emb, map_valid,
        )


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_pred: int,
        use_vmap: bool,
        mlp_head: DictConfig,
        det_anchors: DictConfig, # pos, rot, cls - where & what? (Don't split det and cls since cls important for subsequent traj decoding)
        tf_cfg: DictConfig,
        n_det_layers: int,
        n_traj_layers: int,
        anchor_self_attn: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_pred = n_pred
        self.det_anchors = MultiModalAnchors(
            hidden_dim=hidden_dim, emb_dim=hidden_dim, **det_anchors
        )
        self.n_det = det_anchors["n_pred"]

        self.det2traj_anchors = nn.Linear(in_features=hidden_dim, out_features=hidden_dim * n_pred)

        self.tf_det = TransformerBlock(
            d_model=hidden_dim,
            d_feedforward=hidden_dim * 4,
            n_layer=n_det_layers,
            decoder_self_attn=anchor_self_attn,
            **tf_cfg,
        )

        self.pos_rot_head = nn.Linear(in_features=hidden_dim, out_features=4) # x, y, sin(rot), cos(rot) 
        self.cls_head = nn.Linear(in_features=hidden_dim, out_features=3) # agent types

        self.tf_traj = TransformerBlock(
            d_model=hidden_dim,
            d_feedforward=hidden_dim * 4,
            n_layer=n_traj_layers,
            decoder_self_attn=anchor_self_attn,
            **tf_cfg,
        )

        self.traj_head = MLPHead(hidden_dim=hidden_dim, use_vmap=use_vmap, n_pred=n_pred, **mlp_head)

    def forward(self, n_scene: int, target_type: Tensor, emb: Tensor, emb_invalid: Tensor) -> Tuple[Tensor, Tensor]:
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
        # All valid - hungarian matching later to find best match
        det_valid = torch.ones(n_scene, dtype=torch.bool, device=emb.device)
        det_anchors = self.det_anchors(det_valid, emb, target_type.flatten(0, 1)) # TODO: remove target type
        det_emb, _ = self.tf_det(src=det_anchors, tgt=emb, tgt_padding_mask=emb_invalid)

        pos_rot_pred = self.pos_rot_head(det_emb)
        cls_pred = self.cls_head(det_emb)

        traj_anchors = self.det2traj_anchors(det_emb)
        traj_anchors = rearrange(traj_anchors, "n_scene n_det (hidden_dim n_pred) -> (n_scene n_det) n_pred hidden_dim", n_pred=self.n_pred, hidden_dim=self.hidden_dim)
        emb = repeat(emb, "n_scene n_token hidden_dim -> (n_scene n_det) n_token hidden_dim", n_det=self.n_det)
        emb_invalid = repeat(emb_invalid, "n_scene n_token -> (n_scene n_det) n_token", n_det=self.n_det)

        traj_emb, _ = self.tf_traj(src=traj_anchors, tgt=emb, tgt_padding_mask=emb_invalid)
        traj_emb = rearrange(traj_emb, "(n_scene n_det) n_pred hidden_dim -> n_scene n_det n_pred hidden_dim", n_scene=n_scene)
        traj_emb_valid = torch.ones(traj_emb.shape[:2], dtype=torch.bool, device=traj_emb.device)

        # [n_scene, n_det, n_pred], [n_scene, n_det, n_pred, n_step_future / subsampling_rate, pred_dim]
        traj_conf, traj_pred = self.traj_head(traj_emb_valid, traj_emb, target_type)

        return pos_rot_pred.unsqueeze(0), cls_pred.unsqueeze(0), traj_conf.unsqueeze(0), traj_pred.unsqueeze(0) # n_decoder = 1`