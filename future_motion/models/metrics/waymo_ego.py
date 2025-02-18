from typing import Dict, List, Optional, Tuple
from torch import Tensor

from hptr_modules.models.metrics.waymo import WaymoMetrics


class WaymoEgoMetrics(WaymoMetrics):
    """
    This Loss build upon the WaymoMetrics class. It calculates metrics only for the ego agent.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def filter_ego_batch(
        self,
        batch: Dict[str, Tensor],
        pred_traj: Tensor,
        pred_score: Optional[Tensor] = None,
    ):
        """
        Args:
            batch: Dict[str, Tensor]
                "agent/valid": [n_scene, n_step, n_agent], bool,
                "agent/pos": [n_scene, n_step, n_agent, 2], float32
                "agent/vel": [n_scene, n_step, n_agent, 2], float32, v_x, v_y
                "agent/yaw_bbox": [n_scene, n_step, n_agent, 1], float32, yaw of the bbox heading
                "agent/type": [n_scene, n_agent, 3], bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
                "agent/role": [n_scene, n_agent, 3], bool [sdc=0, interest=1, predict=2]
                "agent/size": [n_scene, n_agent, 3], float32: [length, width, height]
            pred_traj: [n_batch, step_start+1...step_end, n_agent, K, 2]
            pred_score: [n_batch, n_agent, K] normalized prob or None
        Return:
            batch: Dict[str, Tensor]
            pred_traj: [n_batch, step_start+1...step_end, 1, K, 2]
            pred_score: [n_batch, 1, K] normalized prob or None
        """
        agent_role = batch["agent/role"]
        agent_pos = batch["agent/pos"]
        agent_valid = batch["agent/valid"]
        agent_size = batch["agent/size"]
        agent_yaw_bbox = batch["agent/yaw_bbox"]
        agent_vel = batch["agent/vel"]
        agent_type = batch["agent/type"]

        n_scene, n_step, n_agent, _ = agent_pos.shape
        _, n_step_future, _, _, _ = pred_traj.shape

        # (n_scene, n_agent)
        ego_mask = agent_role[..., 0] == True
        ego_mask_n_step = ego_mask.unsqueeze(1).expand(-1, n_step, -1)
        ego_mask_n_step_future = ego_mask.unsqueeze(1).expand(-1, n_step_future, -1)
        assert ego_mask.sum() / n_scene == 1, "Only one ego agent supported"

        ego_batch = {}
        # ! keep ego data; delete all other agents
        ego_batch["agent/valid"] = agent_valid[ego_mask_n_step].view(n_scene, n_step, 1)
        ego_batch["agent/pos"] = agent_pos[ego_mask_n_step].view(n_scene, n_step, 1, 2)
        ego_batch["agent/vel"] = agent_vel[ego_mask_n_step].view(n_scene, n_step, 1, 2)
        ego_batch["agent/yaw_bbox"] = agent_yaw_bbox[ego_mask_n_step].view(
            n_scene, n_step, 1, 1
        )
        ego_batch["agent/role"] = agent_role[ego_mask].view(n_scene, 1, 3)
        ego_batch["agent/type"] = agent_type[ego_mask].view(n_scene, 1, 3)
        ego_batch["agent/size"] = agent_size[ego_mask].view(n_scene, 1, 3)

        ego_pred_traj = pred_traj[ego_mask_n_step_future].view(
            n_scene, n_step_future, 1, -1, 2
        )
        ego_pred_score = (
            pred_score[ego_mask].view(n_scene, 1, -1)
            if pred_score is not None
            else None
        )

        return ego_batch, ego_pred_traj, ego_pred_score

    def forward(
        self,
        batch: Dict[str, Tensor],
        pred_traj: Tensor,
        pred_score: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            pred_traj: [n_batch, step_start+1...step_end, n_agent, K, 2]
            pred_score: [n_batch, n_agent, K] normalized prob or None
        """
        ego_batch, ego_pred_traj, ego_pred_score = self.filter_ego_batch(
            batch, pred_traj, pred_score
        )
        return super().forward(ego_batch, ego_pred_traj, ego_pred_score)
