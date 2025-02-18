import torch

from hptr_modules.data_modules.agent_centric import AgentCentricPreProcessing
from hptr_modules.utils.transform_utils import torch_pos2local, torch_dir2local


class AgentCentricPreProcessingWithEgoNav(AgentCentricPreProcessing):
    def __init__(self, n_route: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_route = n_route

    def forward(self, batch):
        """
        Args: scene-centric Dict
            # map polylines
                "map/on_route": [n_scene, n_pl], bool
            # route (route only for sdc agent)
                "route/valid": (n_scene, n_pl_route, n_pl_node),  # bool
                "route/type": (n_scene, n_pl_route, 11),  # bool one_hot
                "route/pos": (n_scene, n_pl_route, n_pl_node, 2),  # float32
                "route/dir": (n_scene, n_pl_route, n_pl_node, 2),  # float32
                "route/goal": (n_scene, 3)
        Returns: agent-centric Dict, masked according to valid
            # (ref) reference information for transform back to global coordinate and submission to waymo
                "gt/route_valid": [n_scene, n_target, n_pl_route, n_pl_node]
                "gt/route_pos": [n_scene, n_target, n_pl_route, n_pl_node, 2]
                "gt/route_goal": [n_scene, n_target, 2]
                "gt/route_goal_valid": [n_scene, n_target]
                "gt/map_on_route": [n_scene, n_target, n_map, 3]
                "gt/map_valid": [n_scene, n_target, n_map, n_pl_node]
                "gt/map_pos": [n_scene, n_target, n_map, n_pl_node, 2]
            # map polylines
                "ac/map_on_route": [n_scene, n_target, n_map, 3], float32
            # route
                "ac/route_valid": [n_scene, n_target, n_route, n_pl_node], bool
                "ac/route_type": [n_scene, n_target, n_route, 11], one_hot
                "ac/route_pos": [n_scene, n_target, n_route, n_pl_node, 2], float32
                "ac/route_dir": [n_scene, n_target, n_route, n_pl_node, 2], float32
                "ac/route_goal": [n_scene, n_target, 2]
                "ac/route_goal_valid": [n_scene, n_target]
        """
        batch = super().forward(batch)

        # Utils copied from hptr code
        prefix = "" if self.training else "history/"
        n_scene = batch[prefix + "agent/valid"].shape[0]
        ref_pos = batch["ref/pos"]
        ref_rot = batch["ref/rot"]
        # [n_scene, n_pl, n_pl_node, 2], [n_scene, n_target, 1, 2]
        map_dist = torch.norm(batch["map/pos"][:, :, 0].unsqueeze(1) - ref_pos, dim=-1)
        map_dist.masked_fill_(
            ~batch["map/valid"][..., 0].unsqueeze(1), float("inf")
        )  # [n_scene, n_target, n_pl]
        map_dist, map_indices = torch.topk(
            map_dist, self.n_map, largest=False, dim=-1
        )  # [n_scene, n_target, n_map]
        other_scene_indices = torch.arange(n_scene)[:, None, None]  # [n_scene, 1, 1]
        other_target_indices = torch.arange(self.n_target)[
            None, :, None
        ]  # [1, n_target, 1]

        # ! prepare navigation information
        batch["gt/map_valid"] = batch["ac/map_valid"]
        batch["gt/map_pos"] = batch["ac/map_pos"]

        # [n_scene, n_pl] -> [n_scene, n_target, n_map], bool
        ac_map_on_route = (
            batch["map/on_route"]
            .unsqueeze(1)
            .repeat(1, self.n_target, 1)[
                other_scene_indices, other_target_indices, map_indices
            ]
        )
        # [n_scene, n_target, n_map] -> [n_scene, n_target, n_map, 3], float
        # on_route [1,0,0] and not_on_route [0,1,0] for ego navigation and unknown [0,0,1] for other target agents
        device = batch["ref/role"].device
        map_on_route = torch.zeros(
            [n_scene, self.n_target, self.n_map, 3], dtype=torch.float32, device=device
        )
        ego_mask = batch["ref/role"][..., 0].unsqueeze(-1).expand(-1, -1, self.n_map)
        map_on_route[ego_mask & ac_map_on_route] = torch.tensor(
            [1, 0, 0], dtype=torch.float32, device=device
        )
        map_on_route[ego_mask & (~ac_map_on_route)] = torch.tensor(
            [0, 1, 0], dtype=torch.float32, device=device
        )
        map_on_route[~ego_mask] = torch.tensor(
            [0, 0, 1], dtype=torch.float32, device=device
        )
        batch["ac/map_on_route"] = map_on_route
        batch["gt/map_on_route"] = batch["ac/map_on_route"]

        # ! prepare agent-centric route
        # [n_scene, n_pl_route, n_pl_node, 2], [n_scene, n_target, 1, 2]
        route_dist = torch.norm(
            batch["route/pos"][:, :, 0].unsqueeze(1) - ref_pos, dim=-1
        )
        route_dist.masked_fill_(
            ~batch["route/valid"][..., 0].unsqueeze(1), float("inf")
        )  # [n_scene, n_target, n_pl_route]
        route_dist, route_indices = torch.topk(
            route_dist, self.n_route, largest=False, dim=-1
        )  # [n_scene, n_target, n_route]

        # [n_scene, n_pl_route, n_pl_node(20) / n_pl_type(11)] -> [n_scene, n_target, n_route, n_pl_node(20) / n_pl_type(11)]
        for k in ("valid", "type"):
            batch[f"ac/route_{k}"] = (
                batch[f"route/{k}"]
                .unsqueeze(1)
                .repeat(1, self.n_target, 1, 1)[
                    other_scene_indices, other_target_indices, route_indices
                ]
            )
        batch["ac/route_valid"] = batch["ac/route_valid"] & (
            route_dist.unsqueeze(-1) < 3e3
        )
        batch["ac/route_valid"] = (
            batch["ac/route_valid"] & batch["ref/role"][:, :, 0, None, None]
        )  # invalid if not sdc

        # [n_scene, n_pl_route, n_pl_node, 2] -> [n_scene, n_target, n_route, n_pl_node, 2]
        for k in ("pos", "dir"):
            batch[f"ac/route_{k}"] = (
                batch[f"route/{k}"]
                .unsqueeze(1)
                .repeat(1, self.n_target, 1, 1, 1)[
                    other_scene_indices, other_target_indices, route_indices
                ]
            )

        batch["gt/route_valid"] = batch["ac/route_valid"]
        batch["gt/route_pos"] = batch["ac/route_pos"]

        # target_pos: [n_scene, n_target, 1, 2], target_rot: [n_scene, n_target, 2, 2]
        # [n_scene, n_target, n_route, n_pl_node, 2]
        batch["ac/route_pos"] = torch_pos2local(
            batch["ac/route_pos"], ref_pos.unsqueeze(2), ref_rot.unsqueeze(2)
        )
        batch["ac/route_dir"] = torch_dir2local(
            batch["ac/route_dir"], ref_rot.unsqueeze(2)
        )

        batch["ac/route_goal"] = (
            batch["route/goal"]
            .unsqueeze(1)
            .repeat(1, self.n_target, 1)
            .unsqueeze(2)
            .unsqueeze(3)
        )
        batch["ac/route_goal_valid"] = batch["ref/role"][:, :, 0]  # invalid if not sdc
        batch["ac/route_goal"] = (
            torch_pos2local(
                batch["ac/route_goal"][..., :2].float(),
                ref_pos.unsqueeze(2),
                ref_rot.unsqueeze(2),
            )
            .squeeze(2)
            .squeeze(2)
        )
        batch["gt/route_goal"] = batch["ac/route_goal"]
        batch["gt/route_goal_valid"] = batch["ac/route_goal_valid"]

        return batch
