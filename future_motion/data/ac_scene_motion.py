import torch

# from hptr_modules.data_modules.ac_global import AgentCentricGlobal
from future_motion.data.ac_global_ego_nav import AgentCentricWithEgoNavigation


# class AgentCentricSceneMotion(AgentCentricGlobal):
class AgentCentricSceneMotion(AgentCentricWithEgoNavigation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, batch):
        agent_centric_batch = super().forward(batch)
        agent_centric_batch["input/ref_pos"] = batch["ref/pos"]
        agent_centric_batch["input/ref_rot"] = batch["ref/rot"]

        return agent_centric_batch