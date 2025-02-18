import torch

from hptr_modules.utils.pose_pe import PosePE
from hptr_modules.data_modules.ac_global import AgentCentricGlobal


class AgentCentricWithEgoNavigation(AgentCentricGlobal):
    def __init__(
        self,
        pl_aggr_route: bool,
        use_ego_nav: bool,
        nav_with_route: bool,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        data_size = kwargs.get("data_size")
        pose_pe = kwargs.get("pose_pe")

        self.pl_aggr_route = pl_aggr_route
        self.use_ego_nav = use_ego_nav
        self.nav_with_route = nav_with_route

        self.pose_pe_route = PosePE(pose_pe["route"])

        if self.use_ego_nav and self.nav_with_route:
            self.model_kwargs[
                "map_attr_dim"
            ] += 3  # pl on route (on_route, not_on_route, unknown)
        if self.pl_aggr_route:
            route_attr_dim = (
                self.pose_pe_route.out_dim * self.n_pl_node
                + data_size["route/type"][-1]
                + self.n_pl_node
            )
        else:
            route_attr_dim = self.pose_pe_route.out_dim + data_size["route/type"][-1]

        if self.add_ohe:
            if not self.pl_aggr:
                route_attr_dim += self.n_pl_node

        self.model_kwargs["route_attr_dim"] = route_attr_dim
        self.model_kwargs["pl_aggr_route"] = self.pl_aggr_route

    def forward(self, batch):
        """
        Args: agent-centric Dict
            # map polylines
                "ac/map_on_route": [n_scene, n_target, n_map, 3], float32
            # route
                "ac/route_valid": [n_scene, n_target, n_route, n_pl_node], bool
                "ac/route_type": [n_scene, n_target, n_route, 11], bool one_hot
                "ac/route_pos": [n_scene, n_target, n_route, n_pl_node, 2], float32
                "ac/route_dir": [n_scene, n_target, n_route, n_pl_node, 2], float32
                "ac/route_goal": [n_scene, n_target, 2], float32
                "ac/route_goal_valid": [n_scene, n_target], bool
        Returns: add following keys to batch Dict
            # target history, other history, map
                if pl_aggr:
                    "input/route_valid": [n_scene, n_target, n_route], bool
                    "input/route_attr": [n_scene, n_target, n_route, route_attr_dim]
                else:
                    "input/route_valid": [n_scene, n_target, n_route, n_pl_node], bool
                    "input/route_attr": [n_scene, n_target, n_route, n_pl_node, route_attr_dim]
            # route goal
            "input/route_goal_valid": [n_scene, n_target], bool
            "input/route_goal_attr": [n_scene, n_target, 2]

        """
        batch = super().forward(batch)

        valid = batch["ac/target_valid"][:, :, [self.step_current]].unsqueeze(
            -1
        )  # [n_scene, n_target, 1, 1]
        batch["input/route_valid"] = (
            batch["ac/route_valid"] & valid
        )  # [n_scene, n_target, n_route, n_pl_node]
        batch["input/route_goal_valid"] = batch[
            "ac/route_goal_valid"
        ]  # [n_scene, n_target]
        batch["input/route_goal_attr"] = batch["ac/route_goal"]

        # ! randomly mask history target/other/tl
        if self.training and (0 < self.dropout_p_history <= 1.0):
            prob_mask = torch.ones_like(batch["input/route_valid"]) * (
                1 - self.dropout_p_history
            )
            batch["input/route_valid"] &= torch.bernoulli(prob_mask).bool()

        if self.use_ego_nav and self.nav_with_route:
            if self.pl_aggr:
                batch["input/map_attr"] = torch.cat(
                    [
                        batch["input/map_attr"],
                        batch["ac/map_on_route"],  # on route
                    ],
                    dim=-1,
                )
            else:
                batch["input/map_attr"] = torch.cat(
                    [
                        batch["input/map_attr"],
                        batch["ac/map_on_route"]
                        .unsqueeze(-2)
                        .expand(-1, -1, -1, self.n_pl_node, -1),  # on route
                    ],
                    dim=-1,
                )

        # ! prepare "input/route_attr": [n_scene, n_target, n_route, n_pl_node, map_attr_dim]
        if self.pl_aggr_route:  # [n_scene, n_target, n_map, map_attr_dim]
            route_invalid = ~batch["input/route_valid"].unsqueeze(-1)
            route_invalid_reduced = route_invalid.all(-2)
            batch["input/route_attr"] = torch.cat(
                [
                    self.pose_pe_route(batch["ac/route_pos"], batch["ac/route_dir"])
                    .masked_fill(route_invalid, 0)
                    .flatten(-2, -1),
                    batch["ac/route_type"].masked_fill(
                        route_invalid_reduced, 0
                    ),  # n_route_type
                    batch["input/route_valid"],  # n_pl_node
                ],
                dim=-1,
            )
            batch["input/route_valid"] = batch["input/route_valid"].any(
                -1
            )  # [n_scene, n_target, n_route]
        else:  # [n_scene, n_target, n_route, n_pl_node, route_attr_dim]
            batch["input/route_attr"] = torch.cat(
                [
                    self.pose_pe_route(
                        batch["ac/route_pos"], batch["ac/route_dir"]
                    ),  # pl_dim
                    batch["ac/route_type"]
                    .unsqueeze(-2)
                    .expand(-1, -1, -1, self.n_pl_node, -1),  # n_route_type
                ],
                dim=-1,
            )

        if self.add_ohe:
            n_scene, n_target, n_other, _ = batch["ac/other_valid"].shape
            n_route = batch["ac/route_valid"].shape[2]
            if not self.pl_aggr_route:  # there is no need to add ohe if pl_aggr_route
                batch["input/route_attr"] = torch.cat(
                    [
                        batch["input/route_attr"],
                        self.pl_node_ohe[None, None, None, :, :].expand(
                            n_scene, n_target, n_route, -1, -1
                        ),
                    ],
                    dim=-1,
                )
        return batch
