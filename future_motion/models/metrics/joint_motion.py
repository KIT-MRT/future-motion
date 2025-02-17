import torch
import torch.nn.functional as F

from torch import Tensor

from future_motion.models.metrics.barlow_twins import get_barlow_twins_loss


def get_joint_motion_loss(
    motion_emb, env_emb, 
    agent_attr, pred_agent_attr, 
    map_attr, pred_map_attr,
    traffic_light_attr, pred_traffic_light_attr,
    lambda_red, lambda_cme,
    return_mpm_terms: bool = True,
):
    """Computes the JointMotion loss"""
    cme_loss = get_barlow_twins_loss(motion_emb, env_emb, lambda_red)

    agent_loss = F.huber_loss(pred_agent_attr, agent_attr)
    map_loss = F.huber_loss(pred_map_attr, map_attr)
    traffic_light_loss = torch.nan_to_num(F.huber_loss(pred_traffic_light_attr, traffic_light_attr), nan=0.0) # Not all scenes contain traffic lights
    
    mpm_loss = agent_loss + map_loss + traffic_light_loss

    joint_motion_loss = lambda_cme * cme_loss + mpm_loss
    
    if return_mpm_terms:
        return joint_motion_loss, agent_loss, map_loss, traffic_light_loss
    else:
        return joint_motion_loss


def masked_mean_aggregation(token_set: Tensor, mask: Tensor) -> Tensor:
    denominator = torch.sum(mask, -1, keepdim=True)
    feats = torch.sum(token_set * mask.unsqueeze(-1), dim=1) / (denominator + torch.finfo(type=token_set.dtype).eps)
    
    return feats

