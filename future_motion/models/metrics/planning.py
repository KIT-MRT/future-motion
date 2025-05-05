from typing import Dict, Optional
import torch
from torch import Tensor
from torchmetrics.metric import Metric

from hptr_modules.models.metrics.nll import NllMetrics


def gar_loss(mae, alpha, c):
    """
    A General and Adaptive Robust Loss Function
    (see https://openaccess.thecvf.com/content_CVPR_2019/html/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.html).
        For alpha=2, the loss is equivalent to the L2 loss.
        For alpha=1, the loss is equivalent to the Charbonnier loss.
        For alpha=0, the loss is equivalent to the Cauchy loss.
        For alpha=-2, the loss is equivalent to the Geman-McClure loss.
        For alpha=-inf, the loss is equivalent to the Welsch loss.
    :param mae: mean absolute error
    :param alpha: the scale parameter (controls robustness)
    :param c: controls size of loss's quadratic bowl near mae=0
    """
    if alpha == 2:
        raise ValueError("alpha cannot be 2 as it leads to division by zero.")

    abs_a_minus_2 = abs(alpha - 2)
    factor = abs_a_minus_2 / alpha
    term = ((mae / c) ** 2 / abs_a_minus_2 + 1) ** (alpha / 2) - 1
    return factor * term


class EgoPlanningMetrics(NllMetrics):
    """
    This Loss build upon the NllMetrics class and extends it with additional metrics for ego planning.
    """

    def __init__(self, nav_with_route, nav_with_goal, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nav_with_route = nav_with_route
        self.nav_with_goal = False  # nav_with_goal: GoalLoss must be improved with e.g. GAR loss before usage!

        self.ego_loss_prefix = f"{self.prefix}/ego"
        del kwargs["prefix"]

        self.route_loss = RouteLoss(prefix=self.ego_loss_prefix, **kwargs)
        self.goal_loss = GoalLoss(prefix=self.ego_loss_prefix, **kwargs)

        self.add_state("ego_route_loss", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ego_goal_loss", default=torch.tensor(0), dist_reduce_fx="sum")

    def ego_preprocessing(
        self,
        pred_valid: Tensor,
        pred_conf: Tensor,
        pred_pos: Tensor,
        pred_cov: Optional[Tensor],
        ref_role: Tensor,
        ref_type: Tensor,
        gt_valid: Tensor,
        gt_pos: Tensor,
        gt_spd: Tensor,
        gt_vel: Tensor,
        gt_yaw_bbox: Tensor,
        gt_cmd: Tensor,
        gt_route_valid: Tensor,
        gt_route_pos: Tensor,
        gt_route_goal: Tensor,
        gt_route_goal_valid: Tensor,
        gt_map_on_route: Tensor,
        gt_map_valid: Tensor,
        gt_map_pos: Tensor,
        **kwargs,
    ) -> None:
        """
        Args:
            pred_valid: [n_scene, n_agent], bool, n_agent = n_target
            pred_conf: [n_decoder, n_scene, n_agent, n_pred], not normalized!
            pred_pos: [n_decoder, n_scene, n_agent, n_pred, n_step_future, 2]
            pred_cov: [n_decoder, n_scene, n_agent, n_pred, n_step_future, 2, 2]
            ref_role: [n_scene, n_agent, 3], one hot bool [sdc=0, interest=1, predict=2]
            ref_type: [n_scene, n_agent, 3], one hot bool [veh=0, ped=1, cyc=2]
            gt_valid: [n_scene, n_agent, n_step_future], bool
            gt_pos: [n_scene, n_agent, n_step_future, 2]
            gt_spd: [n_scene, n_agent, n_step_future, 1]
            gt_vel: [n_scene, n_agent, n_step_future, 2]
            gt_yaw_bbox: [n_scene, n_agent, n_step_future, 1]
            gt_cmd: [n_scene, n_agent, 8], one hot bool
            gt_route_valid: [n_scene, n_agent, n_pl_route, n_pl_node], bool
            gt_route_pos: [n_scene, n_agent, n_pl_route, n_pl_node, 2]
            gt_route_goal: [n_scene, n_agent, n_pl_route, 2]
        """
        n_decoder, n_scene, n_agent, n_pred, n_step_future, _ = pred_pos.shape
        _, _, n_route, n_pl_node = gt_route_valid.shape

        # ! create ego mask
        ego_mask = ref_role[..., 0] == True
        ego_mask = ego_mask.expand(n_decoder, -1, -1)
        assert (
            ego_mask.sum() / (n_decoder * n_scene) == 1
        ), "Only one ego agent supported"

        ego_batch = {}
        # ! keep ego data; delete all other agents
        ego_batch["pred_valid"] = pred_valid[ego_mask[0]].view(n_scene, 1)
        ego_batch["pred_conf"] = pred_conf[ego_mask].view(n_decoder, n_scene, 1, n_pred)
        ego_batch["pred_pos"] = pred_pos[ego_mask].view(
            n_decoder, n_scene, 1, n_pred, n_step_future, 2
        )
        ego_batch["pred_spd"] = None
        ego_batch["pred_vel"] = None
        ego_batch["pred_yaw_bbox"] = None
        ego_batch["pred_cov"] = pred_cov[ego_mask].view(
            n_decoder, n_scene, 1, n_pred, n_step_future, 2, 2
        )
        ego_batch["ref_role"] = ref_role[ego_mask[0]].view(n_scene, 1, 3)
        ego_batch["ref_type"] = ref_type[ego_mask[0]].view(n_scene, 1, 4)
        ego_batch["gt_valid"] = gt_valid[ego_mask[0]].view(n_scene, 1, n_step_future)
        ego_batch["gt_pos"] = gt_pos[ego_mask[0]].view(n_scene, 1, n_step_future, 2)
        ego_batch["gt_spd"] = gt_spd[ego_mask[0]].view(n_scene, 1, n_step_future, 1)
        ego_batch["gt_vel"] = gt_vel[ego_mask[0]].view(n_scene, 1, n_step_future, 2)
        ego_batch["gt_yaw_bbox"] = gt_yaw_bbox[ego_mask[0]].view(
            n_scene, 1, n_step_future, 1
        )
        ego_batch["gt_cmd"] = gt_cmd[ego_mask[0]].view(n_scene, 1, 8)
        ego_batch["gt_route_valid"] = gt_route_valid[ego_mask[0]].view(
            n_scene, 1, n_route, n_pl_node
        )
        ego_batch["gt_route_pos"] = gt_route_pos[ego_mask[0]].view(
            n_scene, 1, n_route, n_pl_node, 2
        )
        ego_batch["gt_route_goal"] = gt_route_goal[ego_mask[0]].view(n_scene, 1, 2)
        ego_batch["gt_route_goal_valid"] = gt_route_goal_valid[ego_mask[0]].view(
            n_scene, 1, 1
        )
        # [n_scene, n_target, n_map, 3] -> [n_scene, n_map, 3]
        ego_batch["gt_map_on_route"] = gt_map_on_route[ego_mask[0]].view(n_scene, -1, 3)
        # [n_scene, n_target, n_map, n_pl_node] -> [n_scene, n_map, n_pl_node]
        ego_batch["gt_map_valid"] = gt_map_valid[ego_mask[0]].view(
            n_scene, -1, n_pl_node
        )
        # [n_scene, n_target, n_map, n_pl_node, 2] -> [n_scene, n_map, n_pl_node, 2]
        ego_batch["gt_map_pos"] = gt_map_pos[ego_mask[0]].view(
            n_scene, -1, n_pl_node, 2
        )

        return ego_batch

    def forward(self, **kwargs) -> Dict[str, Tensor]:
        loss_dict = super().forward(**kwargs)

        # ! ego preprocessing
        ego_batch = self.ego_preprocessing(**kwargs)

        # ! ego planning loss
        ego_nav_loss = 0
        if self.nav_with_route:
            ego_route_loss_dict = self.route_loss.forward(**ego_batch)
            ego_route_loss = ego_route_loss_dict[f"{self.ego_loss_prefix}/loss"]
            loss_dict[f"{self.ego_loss_prefix}/route_loss"] = ego_route_loss
            ego_nav_loss += ego_route_loss
        if self.nav_with_goal:
            ego_goal_loss_dict = self.goal_loss.forward(**ego_batch)
            ego_goal_loss = ego_goal_loss_dict[f"{self.ego_loss_prefix}/loss"]
            loss_dict[f"{self.ego_loss_prefix}/goal_loss"] = ego_goal_loss
            ego_nav_loss += ego_goal_loss

        ego_planning_loss = ego_nav_loss
        loss_dict[f"{self.ego_loss_prefix}/loss"] = ego_planning_loss
        return loss_dict


class RouteLoss(Metric):
    def __init__(self, prefix: str, n_decoders: int, **kwargs) -> None:
        super().__init__(dist_sync_on_step=False)
        self.prefix = prefix
        self.n_decoders = n_decoders

        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_conf: Tensor,
        pred_pos: Tensor,
        gt_valid: Tensor,
        gt_route_valid: Tensor,
        gt_route_pos: Tensor,
        gt_map_valid: Tensor,
        gt_map_pos: Tensor,
        gt_map_on_route: Tensor,
        **kwargs,
    ) -> None:
        n_decoder, n_scene, n_agent, n_pred, n_step_future, _ = pred_pos.shape
        self.n_scene = n_scene
        self.n_pred = n_pred
        assert n_agent == 1, "Only one ego agent supported"

        # ! prepare avails
        # [n_scene, n_agent, n_step_future]
        avails = gt_valid
        # [n_decoder, n_scene, n_agent, n_step_future]
        avails = avails.unsqueeze(0).expand(n_decoder, -1, -1, -1)
        # [n_decoder, n_scene, n_agent, n_pred, n_step_future]
        avails = avails.unsqueeze(3).expand(-1, -1, -1, n_pred, -1)

        # ! loss for all modes simultaneously, since all ego predictions should be on-route
        # Calculate loss for each decoder and scene separately, because the number of valid
        # route and prediction points can differ between scenes.
        n_map, n_pl_node = gt_map_valid.shape[-2:]
        # [n_scene, n_map, 3] -> [n_scene, n_map, n_pl_node]
        map_on_route_mask = (
            (gt_map_on_route[..., 0] == 1).unsqueeze(-1).expand(-1, -1, n_pl_node)
        )
        map_on_route_and_valid = gt_map_valid & map_on_route_mask
        agent_indices = torch.arange(n_agent).unsqueeze(1).expand(n_agent, n_pred)
        pred_indices = torch.arange(n_pred).unsqueeze(0).expand(n_agent, n_pred)
        distance_to_route_sum = 0
        for i in range(n_decoder):
            for j in range(n_scene):
                if map_on_route_and_valid[j].sum() == 0:
                    return
                # Find the last valid index along n_step_future
                # Using torch.where to find valid indices, then taking the max index per [n_agent, n_pred]
                last_valid_indices = torch.argmax(
                    avails.int()[i, j].flip(dims=[-1]), dim=-1
                )
                last_valid_indices = (
                    n_step_future - 1 - last_valid_indices
                )  # Convert flipped indices to original indices

                final_pred_pos = pred_pos[
                    i, j, agent_indices, pred_indices, last_valid_indices, :
                ].squeeze(0)

                for pos in final_pred_pos:
                    relevant_points_on_route = find_nearest_polyline_point(
                        gt_map_pos[j], map_on_route_and_valid[j], pos
                    )
                    nearest_point = relevant_points_on_route["nearest_point"]
                    nearest_point_predecessor = relevant_points_on_route["predecessor"]
                    nearest_point_successor = relevant_points_on_route["successor"]
                    if (
                        nearest_point_predecessor is None
                        and nearest_point_successor is None
                    ):
                        # TODO: not sure about this
                        print("Only one point in nearest polyline")
                        distance_to_route += torch.sqrt(
                            torch.sum((pos - nearest_point) ** 2)
                        )
                    elif nearest_point_predecessor is None:
                        nearest_point_predecessor = nearest_point
                    elif nearest_point_successor is None:
                        nearest_point_successor = nearest_point
                    distance_to_route = normal_distance_to_line(
                        pos, nearest_point_predecessor, nearest_point_successor
                    )
                    route_gar_loss = gar_loss(distance_to_route, -100, 5)
                    distance_to_route_sum += route_gar_loss
        self.loss = distance_to_route_sum

    def compute(self) -> Dict[str, Tensor]:
        # Compute the average Chamfer loss across all samples
        route_loss = self.loss / (self.n_decoders * self.n_scene * self.n_pred)

        # Return the loss as a dictionary
        out_dict = {f"{self.prefix}/loss": route_loss}
        return out_dict


class GoalLoss(Metric):
    def __init__(self, prefix: str, n_decoders: int, **kwargs) -> None:
        super().__init__(dist_sync_on_step=False)
        self.prefix = prefix
        self.n_decoders = n_decoders

        self.mse_loss = torch.nn.MSELoss()
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_conf: Tensor,
        pred_pos: Tensor,
        gt_valid: Tensor,
        gt_route_goal_valid: Tensor,
        gt_route_goal: Tensor,
        **kwargs,
    ) -> None:
        n_decoder, n_scene, n_agent, n_pred, n_step_future, _ = pred_pos.shape
        self.n_scene = n_scene
        assert n_agent == 1, "Only one ego agent supported"

        # ! prepare avails
        # [n_scene, n_agent, n_step_future]
        avails = gt_valid
        # [n_decoder, n_scene, n_agent, n_step_future]
        avails = avails.unsqueeze(0).expand(n_decoder, -1, -1, -1)
        # [n_decoder, n_scene, n_agent, n_pred, n_step_future]
        avails = avails.unsqueeze(3).expand(-1, -1, -1, n_pred, -1)

        # ! loss for all modes simultaneously, since all ego predictions should be aligned with the goal
        # Calculate loss for each decoder and scene separately, because the number of valid
        # prediction points can differ between scenes.
        goal_mse = 0
        for i in range(n_decoder):
            for j in range(n_scene):
                # (1 2)
                gt_route_goal_scene = gt_route_goal[j]

                # Find the last valid index along n_step_future
                # Using torch.where to find valid indices, then taking the max index per [n_agent, n_pred]
                last_valid_indices = torch.argmax(
                    avails.int()[i, j].flip(dims=[-1]), dim=-1
                )
                last_valid_indices = (
                    n_step_future - 1 - last_valid_indices
                )  # Convert flipped indices to original indices

                # Use the last valid indices to index into the data tensor
                # Gather indices for advanced indexing
                agent_indices = (
                    torch.arange(n_agent).unsqueeze(1).expand(n_agent, n_pred)
                )
                pred_indices = torch.arange(n_pred).unsqueeze(0).expand(n_agent, n_pred)
                final_pred_pos = pred_pos[
                    i, j, agent_indices, pred_indices, last_valid_indices, :
                ]

                goal_mse += self.mse_loss(
                    final_pred_pos,
                    gt_route_goal_scene.unsqueeze(1).expand(n_agent, n_pred, 2),
                )
        self.loss = goal_mse

    def compute(self) -> Dict[str, Tensor]:
        # Compute the MSE loss across all samples
        mse_loss = self.loss / (self.n_decoders * self.n_scene)

        # Return the loss as a dictionary
        out_dict = {f"{self.prefix}/loss": mse_loss}
        return out_dict


def find_nearest_polyline_point(polylines, mask, target_point):
    """
    Find the polyline with the most points closest to a given target point.

    :param polylines: (n_pl, n_points, 2) array of polylines (x, y coordinates)
    :param mask: (n_pl, n_points) array indicating valid points (1 for valid, 0 for invalid)
    :param target_point: A tuple (x, y) representing the target point

    :return: the result dict
    """
    result = {
        "nearest_point": None,
        "predecessor": None,
        "successor": None,
    }

    # Calculate Euclidean distance function
    def euclidean_distance(p1, p2):
        return torch.sqrt(torch.sum((p1 - p2) ** 2, dim=-1))

    # Iterate over each polyline
    best_point_idx = -1
    best_pl_idx = -1
    min_distance = float("inf")
    num_points = -1

    for i, polyline in enumerate(polylines):
        # Get the valid points (where mask is 1)
        valid_points = polyline[mask[i] == 1]
        if valid_points.size(0) == 0:
            continue

        # Calculate distances from all valid points to the target point
        distances = euclidean_distance(valid_points, target_point)

        # Find the index of the point with the smallest distance
        min_dist, min_idx = torch.min(distances, dim=0)

        # Update the best polyline if this one has a closer point
        if min_dist < min_distance:
            min_distance = min_dist
            best_pl_idx = i
            best_point_idx = min_idx.item()
            num_points = valid_points.size(0)

    result["nearest_point"] = polylines[best_pl_idx][mask[best_pl_idx] == 1][
        best_point_idx
    ]
    result["predecessor"] = (
        polylines[best_pl_idx][mask[best_pl_idx] == 1][best_point_idx - 1]
        if best_point_idx > 0
        else None
    )
    result["successor"] = (
        polylines[best_pl_idx][mask[best_pl_idx] == 1][best_point_idx + 1]
        if best_point_idx < num_points - 1
        else None
    )

    return result


def normal_distance_to_line(predicted_point, line_start, line_end):
    """
    Compute the normal (perpendicular) distance from a predicted point to a line defined by two points.

    :param predicted_point: (2,) tensor representing the (x, y) coordinates of the predicted point.
    :param line_start: (2,) tensor representing the (x, y) coordinates of the first point of the line.
    :param line_end: (2,) tensor representing the (x, y) coordinates of the second point of the line.

    :return: A scalar tensor representing the normal distance.
    """
    # Extract coordinates
    x, y = predicted_point[0], predicted_point[1]
    x1, y1 = line_start[0], line_start[1]
    x2, y2 = line_end[0], line_end[1]

    # Compute numerator (absolute area formula)
    numerator = torch.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)

    # Compute denominator (line length)
    denominator = (
        torch.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-8
    )  # Add small epsilon to prevent division by zero

    # Compute distance
    distance = numerator / denominator

    return distance
