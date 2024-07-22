import torch
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from torch import nn, Tensor
from einops import rearrange
from omegaconf import DictConfig

from external_submodules.hptr.src.models.modules.mlp import MLP
from external_submodules.hptr.src.models.modules.point_net import PointNet
from external_submodules.hptr.src.models.modules.transformer import TransformerBlock
from external_submodules.hptr.src.models.modules.multi_modal import MultiModalAnchors
from external_submodules.hptr.src.models.modules.decoder_ensemble import DecoderEnsemble, MLPHead


class Wayformer(nn.Module):
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
        input_projections: DictConfig,
        early_fusion_encoder: DictConfig,
        **kwargs,
    ) -> None:
        super().__init__()
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

        self.encoder = EarlyFusionEncoder(
            hidden_dim=hidden_dim,
            tf_cfg=tf_cfg,
            **early_fusion_encoder
        )

        decoder["tf_cfg"] = tf_cfg
        decoder["hidden_dim"] = hidden_dim
        self.decoder = DecoderEnsemble(n_decoders, decoder_cfg=decoder)

        model_parameters = filter(lambda p: p.requires_grad, self.input_projections.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Input projections parameters: {total_params/1000000:.2f}M")
        model_parameters = filter(lambda p: p.requires_grad, self.encoder.parameters())
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
            target_emb, target_valid, other_emb, other_valid, tl_emb, tl_valid, map_emb, map_valid = self.input_projections(
                target_valid=target_valid,
                target_attr=target_attr,
                other_valid=other_valid,
                other_attr=other_attr,
                map_valid=map_valid,
                map_attr=map_attr,
                tl_valid=tl_valid,
                tl_attr=tl_attr,
            )

            fused_emb, fused_emb_invalid = self.encoder(
                target_emb, target_valid, other_emb, other_valid, tl_emb, tl_valid, map_emb, map_valid, target_type, valid
            )

            conf, pred = self.decoder(valid=valid, target_type=target_type, emb=fused_emb, emb_invalid=fused_emb_invalid)

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
                    n_decoder=n_decoder, n_scene=n_scene, n_target=n_target, n_pred=n_pred, n_step_future=n_step_future, pred_dim=pred_dim,
                )

        assert torch.isfinite(conf).all()
        assert torch.isfinite(pred).all()
        return valid, conf, pred


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
    ) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:
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
            target_emb: [n_batch, 1 or n_step_hist, hidden_dim], n_batch = n_scene * n_target (agent-centric)
            target_valid: [n_batch, 1 or n_step_hist]
            other_emb: [n_batch, n_other or n_other * n_step_hist, hidden_dim]
            other_valid: [n_batch, n_other or n_other * n_step_hist]
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

        if self.add_learned_pe:
            tl_emb = tl_emb + self.pe_tl
            map_emb = map_emb + self.pe_map
            other_emb = other_emb + self.pe_other
            target_emb = target_emb + self.pe_target

        tl_emb = tl_emb.flatten(1, 2)  # [n_batch, n_step_hist * n_tl, :]
        tl_valid = tl_valid.flatten(1, 2)  # [n_batch, n_step_hist * n_tl]
        if self.pl_aggr or self.use_point_net:
            target_emb = target_emb.unsqueeze(1)  # [n_batch, 1, :]
            target_valid = target_valid.unsqueeze(1)  # [n_batch, 1]
        else:
            # target_emb: [n_batch, n_step_hist/1, :], target_valid: [n_batch, n_step_hist/1]
            map_emb = map_emb.flatten(1, 2)  # [n_batch, n_map * n_pl_node, :]
            map_valid = map_valid.flatten(1, 2)  # [n_batch, n_map * n_pl_node]
            other_emb = other_emb.flatten(1, 2)  # [n_batch, n_other * n_step_hist, :]
            other_valid = other_valid.flatten(1, 2)  # [n_batch, n_other * n_step_hist]

        return (
            target_emb, target_valid,
            other_emb, other_valid,
            tl_emb, tl_valid,
            map_emb, map_valid,
        )


class EarlyFusionEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        tf_cfg: DictConfig,
        latent_query: DictConfig,
        n_latent_query: int,
        n_encoder_layers: int,
        use_shared_tf_encoder: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_encoder_layers = n_encoder_layers
        self.n_latent_query = n_latent_query
        self.use_shared_tf_encoder = use_shared_tf_encoder

        if self.n_encoder_layers > 0:
            if self.n_latent_query > 0:
                self.latent_query = MultiModalAnchors(
                    hidden_dim=hidden_dim, emb_dim=hidden_dim, n_pred=self.n_latent_query, **latent_query
                )
                if self.use_shared_tf_encoder:
                    self.tf_latent_query = TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        n_layer=n_encoder_layers,
                        decoder_self_attn=True,
                        **tf_cfg,
                    )
                else:
                    self.tf_latent_cross = TransformerBlock(
                        d_model=hidden_dim, d_feedforward=hidden_dim * 4, n_layer=1, **tf_cfg
                    )
                    self.tf_latent_self = TransformerBlock(
                        d_model=hidden_dim, d_feedforward=hidden_dim * 4, n_layer=n_encoder_layers, **tf_cfg
                    )
            else:
                self.tf_self_attn = TransformerBlock(
                    d_model=hidden_dim, d_feedforward=hidden_dim * 4, n_layer=n_encoder_layers, **tf_cfg
                )

    def forward(
        self,
        target_emb, target_valid,
        other_emb, other_valid,
        tl_emb, tl_valid,
        map_emb, map_valid,
        target_type, valid,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            target_emb: [n_batch, 1 or n_step_hist, hidden_dim], n_batch = n_scene * n_target (agent-centric)
            target_valid: [n_batch, 1 or n_step_hist]
            other_emb: [n_batch, n_other or n_other * n_step_hist, hidden_dim]
            other_valid: [n_batch, n_other or n_other * n_step_hist]
            tl_emb: [n_batch, n_tl * n_step_hist, hidden_dim]
            tl_valid: [n_batch, n_tl * n_step_hist]
            map_emb: [n_batch, n_map or n_map * n_pl_node, hidden_dim]
            map_valid: [n_batch, n_map or n_map * n_pl_node] 
            target_type: [n_scene, n_target, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
            valid: [n_scene, n_target]

        Returns:
            emb: [n_scene * n_target, :, hidden_dim]
            emb_invalid: [n_scene * n_target, :]
        """
        emb = torch.cat([target_emb, other_emb, tl_emb, map_emb], dim=1)
        emb_invalid = ~torch.cat([target_valid, other_valid, tl_valid, map_valid], dim=1)

        if self.n_encoder_layers > 0:
            if self.n_latent_query > 0:
                # [n_scene * n_agent, n_latent_query, out_dim]
                lq_emb = self.latent_query(valid.flatten(0, 1), None, target_type.flatten(0, 1))
                if self.use_shared_tf_encoder:
                    emb, _ = self.tf_latent_query(src=lq_emb, tgt=emb, tgt_padding_mask=emb_invalid)
                else:
                    emb, _ = self.tf_latent_cross(src=lq_emb, tgt=emb, tgt_padding_mask=emb_invalid)
                    emb, _ = self.tf_latent_self(src=emb, tgt=emb)
                emb_invalid = (~valid).flatten(0, 1).unsqueeze(-1).expand(-1, lq_emb.shape[1])
            else:
                emb, _ = self.tf_self_attn(src=emb, tgt=emb, tgt_padding_mask=emb_invalid)

        return emb, emb_invalid
    

class Decoder(nn.Module):
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
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_pred = n_pred
        self.anchors = MultiModalAnchors(
            hidden_dim=hidden_dim, emb_dim=hidden_dim, n_pred=n_pred, **multi_modal_anchors
        )
        self.tf_anchor = TransformerBlock(
            d_model=hidden_dim,
            d_feedforward=hidden_dim * 4,
            n_layer=n_decoder_layers,
            decoder_self_attn=anchor_self_attn,
            **tf_cfg,
        )
        self.mlp_head = MLPHead(hidden_dim=hidden_dim, use_vmap=use_vmap, n_pred=n_pred, **mlp_head)

    def forward(self, valid: Tensor, target_type: Tensor, emb: Tensor, emb_invalid: Tensor) -> Tuple[Tensor, Tensor]:
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
        # [n_batch, n_pred, hidden_dim]
        anchors = self.anchors(valid.flatten(0, 1), emb, target_type.flatten(0, 1))
        # [n_batch, n_pred, hidden_dim]
        emb, _ = self.tf_anchor(src=anchors, tgt=emb, tgt_padding_mask=emb_invalid)
        # [n_scene, n_target, n_pred, hidden_dim]
        emb = emb.view(valid.shape[0], valid.shape[1], self.n_pred, self.hidden_dim)

        conf, pred = self.mlp_head(valid, emb, target_type)

        return conf, pred
