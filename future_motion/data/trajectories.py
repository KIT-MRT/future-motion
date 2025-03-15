import torch


def compute_trajectory_length(traj_batch):
    length = torch.zeros(traj_batch.shape[:-2], device=traj_batch.device)

    for batch_idx, batch in enumerate(traj_batch):
        for traj_idx, traj in enumerate(batch):
            pos = torch.tensor((0.0, 0.0), device=traj_batch.device)[None, ...]
            for step in traj:
                length[batch_idx, traj_idx] += torch.cdist(pos, step[None, ...], p=2)[
                    0, 0
                ]
                pos = step[None, ...]

    return length
