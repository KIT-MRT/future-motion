import torch


class ContrastiveEmbeddings(dict):
    """Class to store contrastive embeddings. Used to train control vectors"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self["speed"] = None  # faster
        self["acceleration"] = None  # accelerating
        self["direction"] = None  # left / right
        self["agent"] = None  # vehicle to pedestrian


def load_contrastive_embed_pairs(embeddings):
    contrastive_pairs = ContrastiveEmbeddings()

    # positive is high, negative is low
    n_spd = min(
        len(embeddings["speed"]["high"].data),
        len(embeddings["speed"]["low"].data),
    )
    h = embeddings["speed"]["high"].data[:n_spd]
    # m = embeddings["speed"]["moderate"].data[:n_spd]
    l = embeddings["speed"]["low"].data[:n_spd]
    contrastive_pairs["speed"] = torch.cat((h, l), dim=0)  # original

    # accelerate is positive, decelerate is negative -> control vec increases acceleration
    n_acc = min(
        len(embeddings["acceleration"]["accelerate"].data),
        len(embeddings["acceleration"]["decelerate"].data),
    )
    a = embeddings["acceleration"]["accelerate"].data[:n_acc]
    d = embeddings["acceleration"]["decelerate"].data[:n_acc]
    contrastive_pairs["acceleration"] = torch.cat((a, d), dim=0)

    # right is positive, left is negative -> leads from left to right control vec
    n_dir = min(
        len(embeddings["direction"]["right"].data),
        len(embeddings["direction"]["left"].data),
    )
    r = embeddings["direction"]["right"].data[:n_dir]
    l = embeddings["direction"]["left"].data[:n_dir]
    contrastive_pairs["direction"] = torch.cat((r, l), dim=0)

    # positive is vehicle, control vector is from pedestrian to vehicle
    n_agt = min(
        len(embeddings["agent"]["vehicle"].data),
        len(embeddings["agent"]["pedestrian"].data),
    )
    v = embeddings["agent"]["vehicle"].data[:n_agt]
    p = embeddings["agent"]["pedestrian"].data[:n_agt]
    contrastive_pairs["agent"] = torch.cat((v, p), dim=0)

    return contrastive_pairs
