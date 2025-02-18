import torch
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from einops import rearrange
from torch import nn, Tensor
from omegaconf import DictConfig

from hptr_modules.models.modules.transformer import TransformerBlock
from hptr_modules.models.modules.decoder_ensemble import DecoderEnsemble

from future_motion.models.ac_wayformer import InputProjections
from future_motion.models.ac_wayformer import InputRouteProjections
from future_motion.models.ac_red_motion import ReductionDecoder


class SceneMotion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        route_attr_dim: int,
        n_pl_node: int,
        use_current_tl: bool,
        pl_aggr: bool,
        pl_aggr_route: bool,
        n_step_hist: int,
        n_decoders: int,
        use_ego_nav: bool,
        nav_with_route: bool,
        nav_with_goal: bool,
        nav_route_late_fusion: bool,
        nav_goal_late_fusion: bool,
        tf_cfg: DictConfig,
        local_encoder: DictConfig,
        local_route_encoder: DictConfig,
        reduction_decoder: DictConfig,
        route_reduction_decoder: DictConfig,
        latent_context_module: DictConfig,
        motion_decoder: DictConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_pred = motion_decoder.n_pred
        self.n_decoders = n_decoders
        self.pl_aggr = pl_aggr
        self.pl_aggr_route = pl_aggr_route
        self.pred_subsampling_rate = kwargs.get("pred_subsampling_rate", 1)
        self.use_ego_nav = use_ego_nav
        self.nav_with_route = nav_with_route
        self.nav_with_goal = nav_with_goal
        self.nav_route_late_fusion = nav_route_late_fusion
        self.nav_goal_late_fusion = nav_goal_late_fusion
        motion_decoder["mlp_head"]["n_step_future"] = motion_decoder["mlp_head"]["n_step_future"] // self.pred_subsampling_rate

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

        self.local_route_encoder = InputRouteProjections(
            hidden_dim=hidden_dim,
            route_attr_dim=route_attr_dim,
            route_goal_attr_dim=2,
            pl_aggr=pl_aggr_route,
            n_pl_node=n_pl_node,
            **local_route_encoder
        )

        # Opt. include in local encoder
        self.to_ref_pos_emb = nn.Linear(2, hidden_dim)
        self.to_ref_rot_emb = nn.Linear(4, hidden_dim)

        self.reduction_decoder = ReductionDecoder(
            hidden_dim=hidden_dim,
            tf_cfg=tf_cfg,
            **reduction_decoder
        )

        self.route_reduction_decoder = ReductionDecoder(
            hidden_dim=hidden_dim,
            tf_cfg=tf_cfg,
            **route_reduction_decoder
        )

        self.latent_context_module = TransformerBlock(
            d_model=hidden_dim, d_feedforward=hidden_dim * 4, **latent_context_module, **tf_cfg
        )

        motion_decoder["tf_cfg"] = tf_cfg
        motion_decoder["hidden_dim"] = hidden_dim
        self.motion_decoder = DecoderEnsemble(n_decoders, decoder_cfg=motion_decoder)

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
        route_valid: Tensor,
        route_attr: Tensor,
        route_goal_valid: Tensor,
        route_goal_attr: Tensor,
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
                    route_valid: [n_scene, n_target, n_route], bool
                    route_attr: [n_scene, n_target, n_route, route_attr_dim]
                else:
                    target_valid: [n_scene, n_target, n_step_hist], bool
                    target_attr: [n_scene, n_target, n_step_hist, agent_attr_dim]
                    other_valid: [n_scene, n_target, n_other, n_step_hist], bool
                    other_attr: [n_scene, n_target, n_other, n_step_hist, agent_attr_dim]
                    map_valid: [n_scene, n_target, n_map, n_pl_node], bool
                    map_attr: [n_scene, n_target, n_map, n_pl_node, map_attr_dim]
                    route_valid: [n_scene, n_target, n_route, n_pl_node], bool
                    route_attr: [n_scene, n_target, n_route, n_pl_node, map_attr_dim]
            # traffic lights: cannot be aggregated, detections are not tracked.
                if use_current_tl:
                    tl_valid: [n_scene, n_target, 1, n_tl], bool
                    tl_attr: [n_scene, n_target, 1, n_tl, tl_attr_dim]
                else:
                    tl_valid: [n_scene, n_target, n_step_hist, n_tl], bool
                    tl_attr: [n_scene, n_target, n_step_hist, n_tl, tl_attr_dim]
            # route goal
            route_goal_valid: [n_scene, n_target], bool
            route_goal_attr: [n_scene, n_target, 2]

        Returns: will be compared to "output/gt_pos": [n_scene, n_agent, n_step_future, 2]
            valid: [n_scene, n_target]
            conf: [n_decoder, n_scene, n_target, n_pred], not normalized!
            pred: [n_decoder, n_scene, n_target, n_pred, n_step_future, pred_dim]
        """
        for _ in range(inference_repeat_n):
            valid = target_valid if self.pl_aggr else target_valid.any(-1)  # [n_scene, n_target]
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

            if self.use_ego_nav:
                route_emb, route_valid, route_goal_emb, route_goal_valid = self.local_route_encoder(
                    route_valid=route_valid,
                    route_attr=route_attr,
                    route_goal_valid=route_goal_valid,
                    route_goal_attr=route_goal_attr,
                )
                if self.nav_with_goal and not self.nav_goal_late_fusion:
                    emb = torch.cat([emb, route_goal_emb], dim=1)
                    emb_invalid = torch.cat([emb_invalid, ~route_goal_valid], dim=1)

            red_emb = self.reduction_decoder(emb=emb, emb_invalid=emb_invalid, valid=valid)

            ref_pos_emb = self.to_ref_pos_emb(kwargs["ref_pos"].flatten(0, 1).flatten(1, 2))
            ref_rot_emb = self.to_ref_rot_emb(kwargs["ref_rot"].flatten(0, 1).flatten(1, 2))
            
            # Concat & rearrange to learn scene-wide context: n_batch = n_scene (not n_scene * n_agent)
            red_emb = torch.cat([red_emb, ref_pos_emb[:, None, :], ref_rot_emb[:, None, :]], dim=1)
            if self.use_ego_nav:
                if self.nav_with_route and self.nav_route_late_fusion:
                    route_red_emb = self.route_reduction_decoder(emb=route_emb, emb_invalid=~route_valid, valid=valid)
                    red_emb = torch.cat([red_emb, route_red_emb], dim=1)
                if self.nav_with_goal and self.nav_goal_late_fusion:
                    red_emb = torch.cat([red_emb, route_goal_emb], dim=1)
            n_scene, n_agent = valid.shape
            scene_emb = rearrange(red_emb, "(n_scene n_agent) n_token ... -> n_scene (n_agent n_token) ...", n_scene=n_scene, n_agent=n_agent, n_token=red_emb.shape[1])
            scene_emb_invalid = torch.zeros(scene_emb.shape[0], scene_emb.shape[1], device=scene_emb.device, dtype=torch.bool)

            scene_emb, _ = self.latent_context_module(src=scene_emb, tgt=scene_emb, tgt_padding_mask=scene_emb_invalid)

            # Rearrange again for motion decoding
            emb = rearrange(scene_emb, "n_scene (n_agent n_token) ... -> (n_scene n_agent) n_token ...", n_scene=n_scene, n_agent=n_agent, n_token=red_emb.shape[1])
            emb_invalid = rearrange(scene_emb_invalid, "n_scene (n_agent n_token) -> (n_scene n_agent) n_token", n_scene=n_scene, n_agent=n_agent)
            conf, pred = self.motion_decoder(valid=valid, target_type=target_type, emb=emb, emb_invalid=emb_invalid)

            if self.pred_subsampling_rate != 1:
                n_decoder, n_scene, n_target, n_pred, _, pred_dim = pred.shape
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
        return valid, conf, pred