import torch
import torch.nn.functional as F

from torch import nn, Tensor
from omegaconf import DictConfig
from einops import repeat, rearrange

from typing import List, Tuple

from external_submodules.hptr.src.models.modules.mlp import MLP
from external_submodules.hptr.src.models.modules.rpe import get_rel_dist
from external_submodules.hptr.src.models.sc_global import SceneCentricGlobal

from .modules.local_attn import LocalEncoder  # opt. rename to LocalAttnBlock / Module

from .metrics.barlow_twins import get_barlow_twins_loss
from .metrics.mae import get_masked_and_unmasked_indices
from .metrics.joint_motion import masked_mean_aggregation, get_joint_motion_loss


class SceneTransformer(SceneCentricGlobal):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        pl_aggr: bool,
        n_tgt_knn: int,
        tf_cfg: DictConfig,
        intra_class_encoder: DictConfig,
        decoder_remove_ego_agent: bool,
        n_decoders: int,
        decoder: DictConfig,
        dist_limit_map: float = 2000,
        dist_limit_tl: float = 2000,
        dist_limit_agent: List[float] = [2000, 2000, 2000],
        **kwargs
    ) -> None:
        super().__init__(
            hidden_dim,
            agent_attr_dim,
            map_attr_dim,
            tl_attr_dim,
            pl_aggr,
            n_tgt_knn,
            tf_cfg,
            intra_class_encoder,
            decoder_remove_ego_agent,
            n_decoders,
            decoder,
            dist_limit_map,
            dist_limit_tl,
            dist_limit_agent,
            **kwargs
        )

        self.pre_training_mode = kwargs.get("pre_training_mode", None)

        if self.pre_training_mode == "joint_motion":
            # Unique mask token per modality
            self.mask_token = nn.Parameter(torch.randn(hidden_dim))
            self.map_mask_token = nn.Parameter(torch.randn(hidden_dim))
            self.tl_mask_token = nn.Parameter(torch.randn(hidden_dim))
            self.hidden_dim = hidden_dim

            self.pre_training_decoder = LocalEncoder(
                n_blocks=3,
                dim=hidden_dim,  # 256
                attn_window=32,
            )

            self.to_position = MLP(
                fc_dims=[hidden_dim, 2], end_layer_activation=False
            )  # rel dist in x and y
            self.to_map_attr = MLP(
                fc_dims=[hidden_dim, 20 * 287], end_layer_activation=False
            )
            self.to_agent_attr = MLP(
                fc_dims=[hidden_dim, 11 * 278], end_layer_activation=False
            )
            self.to_tl_attr = MLP(fc_dims=[hidden_dim, 261], end_layer_activation=False)

            self.env_proj = nn.Sequential(
                nn.Linear(
                    hidden_dim * 2, hidden_dim * 4
                ),  # hidden_dim * 2 since env_emb = concat(map, traffic_lights)
                nn.LayerNorm(hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            self.motion_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.LayerNorm(hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )

        # Opt. plot proj params here.

    def forward(
        self,
        agent_valid: Tensor,
        agent_type: Tensor,
        agent_attr: Tensor,
        agent_pos: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        map_pos: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        tl_pos: Tensor,
        inference_repeat_n: int = 1,
        inference_cache_map: bool = False,
    ) -> Tuple:
        if self.pre_training_mode == "joint_motion":
            dist_limit_agent = 0
            for i in range(agent_type.shape[-1]):  # [n_scene, n_agent]
                dist_limit_agent += agent_type[:, :, i] * self.dist_limit_agent[i]
            dist_limit_agent = dist_limit_agent.unsqueeze(-1)  # [n_scene, n_agent, 1]

            tl_valid = tl_valid.flatten(1, 2)
            tl_attr = tl_attr.flatten(1, 2)
            tl_pos = tl_pos.flatten(1, 2)

            if self.pl_aggr:
                emb_invalid = ~torch.cat([map_valid, tl_valid, agent_valid], dim=1)
            else:
                emb_invalid = ~torch.cat(
                    [map_valid.any(-1), tl_valid, agent_valid.any(-1)], dim=1
                )
            rel_dist = get_rel_dist(
                torch.cat([map_pos, tl_pos, agent_pos], dim=1), emb_invalid
            )

            # Masked polyline modeling (MPM) via boolean valid masks
            n_scene = map_valid.shape[0]
            n_map = map_valid.shape[1]
            n_tl = tl_valid.shape[1]
            n_agent = agent_valid.shape[1]
            batch_range = torch.arange(n_scene, device=map_valid.device)[:, None]

            masked_map_valid = torch.zeros(
                map_valid.shape, dtype=torch.bool, device=map_valid.device
            )
            masked_map_indices, unmasked_map_indices = get_masked_and_unmasked_indices(
                0.6, num_tokens=n_map, device=map_valid.device
            )
            masked_map_valid[batch_range, unmasked_map_indices] = map_valid[
                batch_range, unmasked_map_indices
            ]

            masked_tl_valid = torch.zeros(
                tl_valid.shape, dtype=torch.bool, device=tl_valid.device
            )
            masked_tl_indices, unmasked_tl_indices = get_masked_and_unmasked_indices(
                0.6, num_tokens=n_tl, device=map_valid.device
            )
            masked_tl_valid[batch_range, unmasked_tl_indices] = tl_valid[
                batch_range, unmasked_tl_indices
            ]

            masked_agent_valid = torch.zeros(
                agent_valid.shape, dtype=torch.bool, device=agent_valid.device
            )
            n_hist = agent_valid.shape[2]
            masked_agent_indices, unmasked_agent_indices = (
                get_masked_and_unmasked_indices(
                    0.6, num_tokens=n_hist, device=agent_valid.device
                )
            )
            masked_agent_valid[:, :, unmasked_agent_indices] = agent_valid[
                :, :, unmasked_agent_indices
            ]

            if self.pl_aggr:
                masked_emb_invalid = ~torch.cat(
                    [masked_map_valid, masked_tl_valid, masked_agent_valid], dim=1
                )
            else:
                masked_emb_invalid = ~torch.cat(
                    [
                        masked_map_valid.any(-1),
                        masked_tl_valid,
                        masked_agent_valid.any(-1),
                    ],
                    dim=1,
                )

            masked_rel_dist = get_rel_dist(
                torch.cat([map_pos, tl_pos, agent_pos], dim=1), masked_emb_invalid
            )

            # Encoder
            map_emb, map_emb_valid, tl_emb, tl_emb_valid, agent_emb, agent_emb_valid = (
                self.intra_class_encoder(
                    inference_repeat_n=inference_repeat_n,
                    inference_cache_map=inference_cache_map,
                    agent_valid=masked_agent_valid,
                    agent_attr=agent_attr,
                    map_valid=masked_map_valid,
                    map_attr=map_attr,
                    tl_valid=masked_tl_valid,
                    tl_attr=tl_attr,
                    rel_dist=masked_rel_dist,
                    dist_limit_map=self.dist_limit_map,
                    dist_limit_tl=self.dist_limit_tl,
                    dist_limit_agent=dist_limit_agent,
                )
            )

            # Decoder
            map_mask_tokens = repeat(
                self.map_mask_token,
                "d -> b n d",
                b=n_scene,
                n=masked_map_indices.shape[-1],
            )
            decoder_map_tokens = torch.zeros(map_emb.shape, device=map_emb.device)
            decoder_map_tokens[batch_range, unmasked_map_indices] = map_emb[
                batch_range, unmasked_map_indices
            ]
            decoder_map_tokens[batch_range, masked_map_indices] = map_mask_tokens

            tl_mask_tokens = repeat(
                self.tl_mask_token,
                "d -> b n d",
                b=n_scene,
                n=masked_tl_indices.shape[-1],
            )
            decoder_tl_tokens = torch.zeros(tl_emb.shape, device=tl_emb.device)
            decoder_tl_tokens[batch_range, unmasked_tl_indices] = tl_emb[
                batch_range, unmasked_tl_indices
            ]
            decoder_tl_tokens[batch_range, masked_tl_indices] = tl_mask_tokens

            # Revert masking
            map_emb_valid = map_valid.any(-1)

            agent_emb_valid = agent_valid.any(-1)
            tl_emb_valid = tl_valid

            true_pos = torch.cat([map_pos, tl_pos, agent_pos], dim=1)
            emb_valid = torch.cat([map_emb_valid, tl_emb_valid, agent_emb_valid], dim=1)

            emb = torch.cat([map_emb, tl_emb, agent_emb], dim=1)
            emb = self.pre_training_decoder(x=emb, mask=emb_valid)

            map_emb_mae = emb[:, :n_map]
            tl_emb_mae = emb[:, n_map : n_map + n_tl]
            agent_emb_mae = emb[:, n_map + n_tl :]
            pred_map_attr = self.to_map_attr(map_emb_mae, map_emb_valid)
            pred_map_attr = rearrange(
                pred_map_attr,
                "n_scene n_map (n_node d) -> n_scene n_map n_node d",
                n_node=20,
                d=287,
            )
            pred_tl_attr = self.to_tl_attr(tl_emb_mae, tl_emb_valid)
            pred_agent_attr = self.to_agent_attr(agent_emb_mae, agent_emb_valid)
            pred_agent_attr = rearrange(
                pred_agent_attr,
                "n_scene n_agent (n_time d) -> n_scene n_agent n_time d",
                n_time=11,
                d=278,
            )

            map_emb = masked_mean_aggregation(map_emb, map_emb_valid)
            tl_emb = torch.nan_to_num(
                masked_mean_aggregation(tl_emb, tl_emb_valid), nan=0.0
            )
            agent_emb = masked_mean_aggregation(agent_emb, agent_emb_valid)
            env_emb = self.env_proj(torch.cat((map_emb, tl_emb), dim=-1))
            motion_emb = self.motion_proj(agent_emb)

            loss, agent_loss, map_loss, traffic_light_loss = get_joint_motion_loss(
                motion_emb,
                env_emb,
                agent_attr[agent_valid],
                pred_agent_attr[agent_valid],
                map_attr[map_valid],
                pred_map_attr[map_valid],
                tl_attr[tl_valid],
                pred_tl_attr[tl_valid],
                lambda_red=0.005,
                lambda_cme=0.01,
            )

            return loss, agent_loss, map_loss, traffic_light_loss
        else:
            return super().forward(
                agent_valid,
                agent_type,
                agent_attr,
                agent_pos,
                map_valid,
                map_attr,
                map_pos,
                tl_valid,
                tl_attr,
                tl_pos,
                inference_repeat_n,
                inference_cache_map,
            )
