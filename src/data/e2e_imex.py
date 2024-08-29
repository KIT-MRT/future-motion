import time
import h5py
import torch
import numpy as np

from einops import rearrange

from external_submodules.hptr.src.utils.transform_utils import torch_pos2local
from external_submodules.hptr.src.data_modules.ac_global import AgentCentricGlobal


class EndToEndImEx(AgentCentricGlobal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: make configurable, as params instead of args kwargs
        self.n_pillar_features = 3
        self.bev_height, self.bev_width = 1024, 1024
        self.lidar_range = {
            "x_min": -75, "x_max": 75, "y_min": -75, "y_max": 75, "z_min": -1.25, "z_max": 2.75,
        }
    
    def forward(self, batch):
        agent_centric_batch = super().forward(batch)
        
        # Detection ground truth 1. pos and 2. rot from sdc frame to target agent frame (global2sdc^-1 * global2target)
        batch_size, *_ = batch["ref/idx"].shape
        global_pos = rearrange(batch["agent/pos"], "batch_size n_time n_agent xy -> batch_size n_agent n_time xy")
        global_pos_target = global_pos[torch.arange(batch_size).unsqueeze(1), batch["ref/idx"]]

        is_sdc = batch["ref/role"][:, :, 0]

        # if a scene has no sdc as agent, set sdc to be the first agent in the set and target_valid to false
        # to skip metric computation
        for scene_idx in range(is_sdc.shape[0]):
            if not is_sdc[scene_idx].any(-1):
                is_sdc[scene_idx] = torch.zeros(is_sdc.shape[1], dtype=torch.bool)
                is_sdc[scene_idx, 0] = True
                batch["input/target_valid"][scene_idx] = torch.zeros(batch["input/target_valid"].shape[1:], dtype=torch.bool)
                print(f'{batch["scenario_id"][scene_idx] = } has no SDC')

        sdc_pos = batch["ref/pos"][is_sdc]
        sdc_rot = batch["ref/rot"][is_sdc]
        
        sdc2target_pos = torch_pos2local(global_pos_target[:, :, self.step_current], sdc_pos, sdc_rot)
        sdc2global_rot = sdc_rot.unsqueeze(1).transpose(-1, -2)
        sdc2target_rot = torch.matmul(batch["ref/rot"], sdc2global_rot)

        agent_centric_batch["gt/sdc2target_pos"] = sdc2target_pos
        agent_centric_batch["gt/sdc2target_rot"] = sdc2target_rot
        agent_centric_batch["gt/target_type"] = agent_centric_batch["input/target_type"]

        
        lidar_pillars = torch.zeros(
            size=(batch_size, self.n_step_hist, self.n_pillar_features, self.bev_height, self.bev_width),
            device=batch["ref/idx"].device,
        )
        
        for batch_idx in range(batch_size):
            for step_hist in range(self.n_step_hist):
                lidar_pillars[batch_idx, step_hist] = batch[f"lidar_pillars/{step_hist}"][batch_idx]

        agent_centric_batch["input/lidar_pillars"] = lidar_pillars

        # map to only sdcs map, remove others past motion (history), romve other targets besides the sdc, target type, etc.
        agent_centric_batch["input/map_attr"] = agent_centric_batch["input/map_attr"][is_sdc].unsqueeze(1)
        agent_centric_batch["input/map_valid"] = agent_centric_batch["input/map_valid"][is_sdc].unsqueeze(1)
        agent_centric_batch["input/tl_attr"] = agent_centric_batch["input/tl_attr"][is_sdc].unsqueeze(1)
        agent_centric_batch["input/tl_valid"] = agent_centric_batch["input/tl_valid"][is_sdc].unsqueeze(1)
        
        agent_centric_batch["input/sdc_valid"] = agent_centric_batch["input/target_valid"][is_sdc].unsqueeze(1) # Prob. always valid (or just not for rare GNSS/GPS jumps)
        agent_centric_batch["input/sdc_type"] = agent_centric_batch["input/target_type"][is_sdc].unsqueeze(1) # Dummy value to be consistent with other models as always type = vehicle
        agent_centric_batch["input/sdc_attr"] = agent_centric_batch["input/target_attr"][is_sdc].unsqueeze(1)

        # Remove non-processed feats
        # del agent_centric_batch["input/target_valid"]
        # del agent_centric_batch["input/target_type"]
        del agent_centric_batch["input/target_attr"]
        del agent_centric_batch["input/other_valid"]
        del agent_centric_batch["input/other_attr"]
        
        return agent_centric_batch