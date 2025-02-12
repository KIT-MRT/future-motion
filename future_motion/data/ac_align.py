import torch

from typing import Dict
from torch import nn, Tensor
from omegaconf import DictConfig
from hptr.utils.pose_pe import PosePE

from einops import rearrange
from einops.layers.torch import Rearrange

from future_motion.data.lang_labels import get_label_id2
from future_motion.data.lang_labels import get_text_description
from future_motion.data.lang_labels import agent_dict, direction_dict, speed_dict, acceleration_dict
from future_motion.data.lang_labels import get_speed_class, get_acceleration_class, get_label_id2, classify_movement



class AgentCentricAlign(nn.Module):
    def __init__(
        self,
        time_step_current: int,
        data_size: DictConfig,
        dropout_p_history: float,
        use_current_tl: bool,
        add_ohe: bool,
        pl_aggr: bool,
        pose_pe: DictConfig,
        describe_current_speed: bool = False,
    ) -> None:
        super().__init__()
        self.n_scene_n_target_to_n_batch = Rearrange("n_scene n_target ... -> (n_scene n_target) ...")
        self.dropout_p_history = dropout_p_history  # [0, 1], turn off if set to negative
        self.step_current = time_step_current
        self.n_step_hist = time_step_current + 1
        self.use_current_tl = use_current_tl
        self.add_ohe = add_ohe
        self.pl_aggr = pl_aggr
        self.n_pl_node = data_size["map/valid"][-1]

        self.pose_pe_agent = PosePE(pose_pe["agent"])
        self.pose_pe_map = PosePE(pose_pe["map"])
        self.pose_pe_tl = PosePE(pose_pe["tl"])

        self.describe_current_speed = describe_current_speed
        print(f"{self.describe_current_speed = }")

        tl_attr_dim = self.pose_pe_tl.out_dim + data_size["tl_stop/state"][-1]
        if self.pl_aggr:
            agent_attr_dim = (
                self.pose_pe_agent.out_dim * self.n_step_hist
                + data_size["agent/spd"][-1] * self.n_step_hist  # 1
                + data_size["agent/vel"][-1] * self.n_step_hist  # 2
                + data_size["agent/yaw_rate"][-1] * self.n_step_hist  # 1
                + data_size["agent/acc"][-1] * self.n_step_hist  # 1
                + data_size["agent/size"][-1]  # 3
                + data_size["agent/type"][-1]  # 3
                + self.n_step_hist  # valid
            )
            map_attr_dim = self.pose_pe_map.out_dim * self.n_pl_node + data_size["map/type"][-1] + self.n_pl_node
        else:
            agent_attr_dim = (
                self.pose_pe_agent.out_dim
                + data_size["agent/spd"][-1]  # 1
                + data_size["agent/vel"][-1]  # 2
                + data_size["agent/yaw_rate"][-1]  # 1
                + data_size["agent/acc"][-1]  # 1
                + data_size["agent/size"][-1]  # 3
                + data_size["agent/type"][-1]  # 3
            )
            map_attr_dim = self.pose_pe_map.out_dim + data_size["map/type"][-1]

        if self.add_ohe:
            self.register_buffer("history_step_ohe", torch.eye(self.n_step_hist))
            self.register_buffer("pl_node_ohe", torch.eye(self.n_pl_node))
            if not self.pl_aggr:
                map_attr_dim += self.n_pl_node
                agent_attr_dim += self.n_step_hist
            if not self.use_current_tl:
                tl_attr_dim += self.n_step_hist

        self.model_kwargs = {
            "agent_attr_dim": agent_attr_dim,
            "map_attr_dim": map_attr_dim,
            "tl_attr_dim": tl_attr_dim,
            "n_step_hist": self.n_step_hist,
            "n_pl_node": self.n_pl_node,
            "use_current_tl": self.use_current_tl,
            "pl_aggr": self.pl_aggr,
        }

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args: agent-centric Dict
            # (ref) reference information for transform back to global coordinate and submission to waymo
                "ref/pos": [n_scene, n_target, 1, 2]
                "ref/rot": [n_scene, n_target, 2, 2]
                "ref/idx": [n_scene, n_target]
                "ref/idx_n": int, original number of agents
                "ref/role": [n_scene, n_target, 3]
                "ref/type": [n_scene, n_target, 3]
            # (gt) ground-truth target future for training, not available for testing
                "gt/valid": [n_scene, n_target, n_step_future], bool
                "gt/pos": [n_scene, n_target, n_step_future, 2]
                "gt/spd": [n_scene, n_target, n_step_future, 1]
                "gt/vel": [n_scene, n_target, n_step_future, 2]
                "gt/yaw_bbox": [n_scene, n_target, n_step_future, 1]
                "gt/cmd": [n_scene, n_target, 8]
            # (ac) agent-centric target agents states
                "ac/target_valid": [n_scene, n_target, n_step_hist]
                "ac/target_pos": [n_scene, n_target, n_step_hist, 2]
                "ac/target_vel": [n_scene, n_target, n_step_hist, 2]
                "ac/target_spd": [n_scene, n_target, n_step_hist, 1]
                "ac/target_acc": [n_scene, n_target, n_step_hist, 1]
                "ac/target_yaw_bbox": [n_scene, n_target, n_step_hist, 1]
                "ac/target_yaw_rate": [n_scene, n_target, n_step_hist, 1]
            # target agents attributes
                "ac/target_type": [n_scene, n_target, 3]
                "ac/target_role": [n_scene, n_target, 3]
                "ac/target_size": [n_scene, n_target, 3]
            # other agents states
                "ac/other_valid": [n_scene, n_target, n_other, n_step_hist]
                "ac/other_pos": [n_scene, n_target, n_other, n_step_hist, 2]
                "ac/other_vel": [n_scene, n_target, n_other, n_step_hist, 2]
                "ac/other_spd": [n_scene, n_target, n_other, n_step_hist, 1]
                "ac/other_acc": [n_scene, n_target, n_other, n_step_hist, 1]
                "ac/other_yaw_bbox": [n_scene, n_target, n_other, n_step_hist, 1]
                "ac/other_yaw_rate": [n_scene, n_target, n_other, n_step_hist, 1]
            # other agents attributes
                "ac/other_type": [n_scene, n_target, n_other, 3]
                "ac/other_role": [n_scene, n_target, n_other, 3]
                "ac/other_size": [n_scene, n_target, n_other, 3]
            # map polylines
                "ac/map_valid": [n_scene, n_target, n_map, n_pl_node], bool
                "ac/map_type": [n_scene, n_target, n_map, 11], bool one_hot
                "ac/map_pos": [n_scene, n_target, n_map, n_pl_node, 2], float32
                "ac/map_dir": [n_scene, n_target, n_map, n_pl_node, 2], float32
            # traffic lights
                "ac/tl_valid": [n_scene, n_target, n_step_hist, n_tl], bool
                "ac/tl_state": [n_scene, n_target, n_step_hist, n_tl, 5], bool one_hot
                "ac/tl_pos": [n_scene, n_target, n_step_hist, n_tl, 2], x,y
                "ac/tl_dir": [n_scene, n_target, n_step_hist, n_tl, 2], x,y

        Returns: add following keys to batch Dict
            # target type: no need to be aggregated.
                "input/target_type": [n_scene, n_target, 3]
            # target history, other history, map
                if pl_aggr:
                    "input/target_valid": [n_scene, n_target], bool
                    "input/target_attr": [n_scene, n_target, agent_attr_dim]
                    "input/other_valid": [n_scene, n_target, n_other], bool
                    "input/other_attr": [n_scene, n_target, n_other, agent_attr_dim]
                    "input/map_valid": [n_scene, n_target, n_map], bool
                    "input/map_attr": [n_scene, n_target, n_map, map_attr_dim]
                else:
                    "input/target_valid": [n_scene, n_target, n_step_hist], bool
                    "input/target_attr": [n_scene, n_target, n_step_hist, agent_attr_dim]
                    "input/other_valid": [n_scene, n_target, n_other, n_step_hist], bool
                    "input/other_attr": [n_scene, n_target, n_other, n_step_hist, agent_attr_dim]
                    "input/map_valid": [n_scene, n_target, n_map, n_pl_node], bool
                    "input/map_attr": [n_scene, n_target, n_map, n_pl_node, map_attr_dim]
            # traffic lights: stop point, cannot be aggregated, detections are not tracked, singular node polyline.
                if use_current_tl:
                    "input/tl_valid": [n_scene, n_target, 1, n_tl], bool
                    "input/tl_attr": [n_scene, n_target, 1, n_tl, tl_attr_dim]
                else:
                    "input/tl_valid": [n_scene, n_target, n_step_hist, n_tl], bool
                    "input/tl_attr": [n_scene, n_target, n_step_hist, n_tl, tl_attr_dim]
        """
        batch["input/target_type"] = batch["ac/target_type"]
        valid = batch["ac/target_valid"][:, :, [self.step_current]].unsqueeze(-1)  # [n_scene, n_target, 1, 1]
        batch["input/target_valid"] = batch["ac/target_valid"]  # [n_scene, n_target, n_step_hist]
        batch["input/other_valid"] = batch["ac/other_valid"] & valid  # [n_scene, n_target, n_other, n_step_hist]
        batch["input/tl_valid"] = batch["ac/tl_valid"] & valid  # [n_scene, n_target, n_step_hist, n_tl]
        batch["input/map_valid"] = batch["ac/map_valid"] & valid  # [n_scene, n_target, n_map, n_pl_node]

        # ! randomly mask history target/other/tl
        if self.training and (0 < self.dropout_p_history <= 1.0):
            prob_mask = torch.ones_like(batch["input/target_valid"][..., :-1]) * (1 - self.dropout_p_history)
            batch["input/target_valid"][..., :-1] &= torch.bernoulli(prob_mask).bool()
            prob_mask = torch.ones_like(batch["input/other_valid"]) * (1 - self.dropout_p_history)
            batch["input/other_valid"] &= torch.bernoulli(prob_mask).bool()
            prob_mask = torch.ones_like(batch["input/tl_valid"]) * (1 - self.dropout_p_history)
            batch["input/tl_valid"] &= torch.bernoulli(prob_mask).bool()
            prob_mask = torch.ones_like(batch["input/map_valid"]) * (1 - self.dropout_p_history)
            batch["input/map_valid"] &= torch.bernoulli(prob_mask).bool()

        # ! prepare "input/target_attr"
        if self.pl_aggr:  # [n_scene, n_target, agent_attr_dim]
            target_invalid = ~batch["input/target_valid"].unsqueeze(-1)  # [n_scene, n_target, n_step_hist, 1]
            target_invalid_reduced = target_invalid.all(-2)  # [n_scene, n_target, 1]
            batch["input/target_attr"] = torch.cat(
                [
                    self.pose_pe_agent(batch["ac/target_pos"], batch["ac/target_yaw_bbox"])
                    .masked_fill(target_invalid, 0)
                    .flatten(-2, -1),
                    batch["ac/target_vel"].masked_fill(target_invalid, 0).flatten(-2, -1),  # n_step_hist*2
                    batch["ac/target_spd"].masked_fill(target_invalid, 0).squeeze(-1),  # n_step_hist
                    batch["ac/target_yaw_rate"].masked_fill(target_invalid, 0).squeeze(-1),  # n_step_hist
                    batch["ac/target_acc"].masked_fill(target_invalid, 0).squeeze(-1),  # n_step_hist
                    batch["ac/target_size"].masked_fill(target_invalid_reduced, 0),  # 3
                    batch["ac/target_type"].masked_fill(target_invalid_reduced, 0),  # 3
                    batch["input/target_valid"],  # n_step_hist
                ],
                dim=-1,
            )
            batch["input/target_valid"] = batch["input/target_valid"].any(-1)  # [n_scene, n_target]
        else:  # [n_scene, n_target, n_step_hist, agent_attr_dim]
            batch["input/target_attr"] = torch.cat(
                [
                    self.pose_pe_agent(batch["ac/target_pos"], batch["ac/target_yaw_bbox"]),
                    batch["ac/target_vel"],  # vel xy, 2
                    batch["ac/target_spd"],  # speed, 1
                    batch["ac/target_yaw_rate"],  # yaw rate, 1
                    batch["ac/target_acc"],  # acc, 1
                    batch["ac/target_size"].unsqueeze(-2).expand(-1, -1, self.n_step_hist, -1),  # 3
                    batch["ac/target_type"].unsqueeze(-2).expand(-1, -1, self.n_step_hist, -1),  # 3
                ],
                dim=-1,
            )

        # ! prepare "input/other_attr"
        if self.pl_aggr:  # [n_scene, n_target, n_other, agent_attr_dim]
            other_invalid = ~batch["input/other_valid"].unsqueeze(-1)
            other_invalid_reduced = other_invalid.all(-2)
            batch["input/other_attr"] = torch.cat(
                [
                    self.pose_pe_agent(batch["ac/other_pos"], batch["ac/other_yaw_bbox"])
                    .masked_fill(other_invalid, 0)
                    .flatten(-2, -1),
                    batch["ac/other_vel"].masked_fill(other_invalid, 0).flatten(-2, -1),  # n_step_hist*2
                    batch["ac/other_spd"].masked_fill(other_invalid, 0).squeeze(-1),  # n_step_hist
                    batch["ac/other_yaw_rate"].masked_fill(other_invalid, 0).squeeze(-1),  # n_step_hist
                    batch["ac/other_acc"].masked_fill(other_invalid, 0).squeeze(-1),  # n_step_hist
                    batch["ac/other_size"].masked_fill(other_invalid_reduced, 0),  # 3
                    batch["ac/other_type"].masked_fill(other_invalid_reduced, 0),  # 3
                    batch["input/other_valid"],  # n_step_hist
                ],
                dim=-1,
            )
            batch["input/other_valid"] = batch["input/other_valid"].any(-1)  # [n_scene, n_target, n_other]
        else:  # [n_scene, n_target, n_other, n_step_hist, agent_attr_dim]
            batch["input/other_attr"] = torch.cat(
                [
                    self.pose_pe_agent(batch["ac/other_pos"], batch["ac/other_yaw_bbox"]),
                    batch["ac/other_vel"],  # vel xy, 2
                    batch["ac/other_spd"],  # speed, 1
                    batch["ac/other_yaw_rate"],  # yaw rate, 1
                    batch["ac/other_acc"],  # acc, 1
                    batch["ac/other_size"].unsqueeze(-2).expand(-1, -1, -1, self.n_step_hist, -1),  # 3
                    batch["ac/other_type"].unsqueeze(-2).expand(-1, -1, -1, self.n_step_hist, -1),  # 3
                ],
                dim=-1,
            )

        # ! prepare "input/map_attr": [n_scene, n_target, n_map, n_pl_node, map_attr_dim]
        if self.pl_aggr:  # [n_scene, n_target, n_map, map_attr_dim]
            map_invalid = ~batch["input/map_valid"].unsqueeze(-1)
            map_invalid_reduced = map_invalid.all(-2)
            batch["input/map_attr"] = torch.cat(
                [
                    self.pose_pe_map(batch["ac/map_pos"], batch["ac/map_dir"])
                    .masked_fill(map_invalid, 0)
                    .flatten(-2, -1),
                    batch["ac/map_type"].masked_fill(map_invalid_reduced, 0),  # n_map_type
                    batch["input/map_valid"],  # n_pl_node
                ],
                dim=-1,
            )
            batch["input/map_valid"] = batch["input/map_valid"].any(-1)  # [n_scene, n_target, n_map]
        else:  # [n_scene, n_target, n_map, n_pl_node, map_attr_dim]
            batch["input/map_attr"] = torch.cat(
                [
                    self.pose_pe_map(batch["ac/map_pos"], batch["ac/map_dir"]),  # pl_dim
                    batch["ac/map_type"].unsqueeze(-2).expand(-1, -1, -1, self.n_pl_node, -1),  # n_map_type
                ],
                dim=-1,
            )

        # ! prepare "input/tl_attr": [n_scene, n_target, n_step_hist/1, n_tl, tl_attr_dim]
        # [n_scene, n_target, n_step_hist, n_tl, 2]
        tl_pos = batch["ac/tl_pos"]
        tl_dir = batch["ac/tl_dir"]
        tl_state = batch["ac/tl_state"]
        if self.use_current_tl:
            tl_pos = tl_pos[:, :, [-1]]  # [n_scene, n_target, 1, n_tl, 2]
            tl_dir = tl_dir[:, :, [-1]]  # [n_scene, n_target, 1, n_tl, 2]
            tl_state = tl_state[:, :, [-1]]  # [n_scene, n_target, 1, n_tl, 5]
            batch["input/tl_valid"] = batch["input/tl_valid"][:, :, [-1]]  # [n_scene, n_target, 1, n_tl]
        batch["input/tl_attr"] = torch.cat([self.pose_pe_tl(tl_pos, tl_dir), tl_state], dim=-1)

        # ! add one-hot encoding for sequence (temporal, order of polyline nodes)
        if self.add_ohe:
            n_scene, n_target, n_other, _ = batch["ac/other_valid"].shape
            n_map = batch["ac/map_valid"].shape[2]
            if not self.pl_aggr:  # there is no need to add ohe if pl_aggr
                batch["input/target_attr"] = torch.cat(
                    [
                        batch["input/target_attr"],
                        self.history_step_ohe[None, None, :, :].expand(n_scene, n_target, -1, -1),
                    ],
                    dim=-1,
                )
                batch["input/other_attr"] = torch.cat(
                    [
                        batch["input/other_attr"],
                        self.history_step_ohe[None, None, None, :, :].expand(n_scene, n_target, n_other, -1, -1),
                    ],
                    dim=-1,
                )
                batch["input/map_attr"] = torch.cat(
                    [
                        batch["input/map_attr"],
                        self.pl_node_ohe[None, None, None, :, :].expand(n_scene, n_target, n_map, -1, -1),
                    ],
                    dim=-1,
                )

            if not self.use_current_tl:  # there is no need to add ohe if use_current_tl
                n_tl = batch["input/tl_valid"].shape[-1]
                batch["input/tl_attr"] = torch.cat(
                    [
                        batch["input/tl_attr"],
                        self.history_step_ohe[None, None, :, None, :].expand(n_scene, n_target, -1, n_tl, -1),
                    ],
                    dim=-1,
                )

        n_scene, n_target = batch["ac/target_type"].shape[:2]

        mask = batch["ac/target_role"][..., 0] + batch["ac/target_role"][..., 1] + batch["ac/target_role"][..., 2]

        typ_ = self.n_scene_n_target_to_n_batch(batch["ac/target_type"])
        yaw_ = self.n_scene_n_target_to_n_batch(batch["ac/target_yaw_rate"])
        spd_ = self.n_scene_n_target_to_n_batch(batch["ac/target_spd"])
        siz_ = self.n_scene_n_target_to_n_batch(batch["ac/target_size"])
        msk_ = self.n_scene_n_target_to_n_batch(mask)

        batch_size = typ_.shape[0]

        text = []
        l_dir_ids = []
        l_label_ids = []
        l_agent_ids = []
        l_spd_ids = []
        l_acc_ids = []

        for i in range(batch_size):

            if not typ_[i].any():
                text.append("undefined")
                l_dir_ids.append(-1)
                l_label_ids.append(-1)
                l_agent_ids.append(-1)
                l_spd_ids.append(-1)
                l_acc_ids.append(-1)

                continue
            
            k = typ_[i].nonzero()[0].item()
            agent_type = ["vehicle", "pedestrian", "cyclist"][k]

            dir_class = classify_movement(yaw_[i], spd_[i])
            spd_class = get_speed_class(spd_[i])
            acc_class = get_acceleration_class(spd_[i])

            l_dir_ids.append(direction_dict[dir_class])
            l_agent_ids.append(agent_dict[agent_type]) # can prob. be simplified (see above)
            l_spd_ids.append(speed_dict[spd_class])
            l_acc_ids.append(acceleration_dict[acc_class])

            # motion label
            l_label_ids.append(get_label_id2(agent_type, dir_class, spd_class, acc_class))

        batch["input/agent_mask"] = msk_
        batch["input/direction_labels"] = torch.tensor(l_dir_ids)
        batch["input/agent_labels"] = torch.tensor(l_agent_ids)
        batch["input/speed_labels"] = torch.tensor(l_spd_ids)
        batch["input/acceleration_labels"] = torch.tensor(l_acc_ids)

        batch["input/motion_labels"] = torch.tensor(l_label_ids)

        return batch
