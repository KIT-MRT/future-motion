import time
import torch
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from typing import Dict, Optional

from scipy.optimize import linear_sum_assignment

from external_submodules.hptr.src.models.metrics.nll import NllMetrics


class HungarianNLL(NllMetrics):
    full_state_update = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("error_det_pos_rot", default=torch.tensor(0.0), dist_reduce_fx="sum") 
        self.add_state("error_det_cls", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_det_pos_rot: Tensor,
        pred_det_cls: Tensor,
        pred_valid: Tensor,
        pred_conf: Tensor,
        pred_pos: Tensor,
        pred_spd: Optional[Tensor],
        pred_vel: Optional[Tensor],
        pred_yaw_bbox: Optional[Tensor],
        pred_cov: Optional[Tensor],
        ref_role: Tensor,
        ref_type: Tensor,
        gt_sdc2target_pos: Tensor,
        gt_sdc2target_rot: Tensor,
        gt_target_type: Tensor,
        gt_valid: Tensor,
        gt_pos: Tensor,
        gt_spd: Tensor,
        gt_vel: Tensor,
        gt_yaw_bbox: Tensor,
        gt_cmd: Tensor,
        **kwargs,
    ):
        # Hungarian matching before update
        gt_det_rot = gt_sdc2target_rot[:, :, 1] # sin, cos

        gt_det_pos_rot = torch.cat((gt_sdc2target_pos, gt_det_rot), dim=-1)
        pred_det_pos_rot = pred_det_pos_rot[0] # Assume n_decoder = 1

        # Match per scene, not across multiple scenes
        n_scene = gt_det_pos_rot.shape[0]
        pred_idx = torch.zeros(gt_det_pos_rot.shape[:2], dtype=torch.long)

        for scene_idx in range(n_scene):
            pred_idx[scene_idx] = sorted_hungarian_matching(pred_det_pos_rot[scene_idx], gt_det_pos_rot[scene_idx])

        matched_pred_det_pos_rot = pred_det_pos_rot[torch.arange(n_scene).unsqueeze(1), pred_idx]

        self.error_det_pos_rot = F.l1_loss(matched_pred_det_pos_rot, gt_det_pos_rot)

        pred_det_cls = pred_det_cls[0]
        gt_det_cls = gt_target_type.to(dtype=torch.int).argmax(dim=-1)
        matched_pred_det_cls = pred_det_cls[torch.arange(n_scene).unsqueeze(1), pred_idx]

        matched_gt_det_cls = rearrange(gt_det_cls, "n_scene n_target -> (n_scene n_target)")
        matched_pred_det_cls = rearrange(matched_pred_det_cls, "n_scene n_target n_class -> (n_scene n_target) n_class")
        self.error_det_cls = F.cross_entropy(matched_pred_det_cls, matched_gt_det_cls)

        pred_conf = pred_conf[0]
        pred_pos = pred_pos[0]
        pred_cov = pred_cov[0]

        matched_pred_conf = pred_conf[torch.arange(n_scene).unsqueeze(1), pred_idx] 
        matched_pred_pos = pred_pos[torch.arange(n_scene).unsqueeze(1), pred_idx]
        matched_pred_cov = pred_cov[torch.arange(n_scene).unsqueeze(1), pred_idx]

        super().update(
            pred_valid=pred_valid,
            pred_conf=matched_pred_conf.unsqueeze(0), # n_decoder = 1
            pred_pos=matched_pred_pos.unsqueeze(0),
            pred_cov=matched_pred_cov.unsqueeze(0),
            pred_spd=pred_spd,
            pred_vel=pred_vel,
            pred_yaw_bbox=pred_yaw_bbox,
            ref_role=ref_role,
            ref_type=ref_type,
            gt_valid=gt_valid,
            gt_pos=gt_pos,
            gt_spd=gt_spd,
            gt_vel=gt_vel,
            gt_yaw_bbox=gt_yaw_bbox,
            gt_cmd=gt_cmd,
            **kwargs
        )
    

    def compute(self):
        out_dict = super().compute()
        out_dict[f"{self.prefix}/error_det_pos_rot"] = self.error_det_pos_rot
        out_dict[f"{self.prefix}/error_det_cls"] = self.error_det_cls

        out_dict[f"{self.prefix}/loss"] += self.error_det_pos_rot + self.error_det_cls # TODO: add loss weights as hyperparams

        return out_dict


def sorted_hungarian_matching(preds, targets):
    cost_matrix = torch.cdist(preds, targets, p=1)
    preds_idx, targets_idx = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    preds_idx, targets_idx = torch.tensor(preds_idx, dtype=torch.long), torch.tensor(targets_idx, dtype=torch.long)
    
    # Argsort targets_idx and reorder preds_idx to skip changing the order of targets afterwards and simplify the code
    return preds_idx[targets_idx.argsort()]