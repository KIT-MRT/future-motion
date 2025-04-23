import torch
from einops import rearrange
from tqdm import tqdm


# Define how each integer label maps to the corresponding dictionary key
label_map = {
    "speed": {0: "low", 1: "moderate", 2: "high", 3: "backwards"},
    "acceleration": {0: "decelerate", 1: "constant", 2: "accelerate"},
    "direction": {0: "stationary", 1: "straight", 2: "right", 3: "left"},
    "agent": {0: "vehicle", 1: "pedestrian", 2: "cyclist"},
}


class EmbeddingData:
    """
    Container class to store embeddings
    """

    def __init__(self):
        self.data = []
        self._mean = None
        self._var = None

    @property
    def mean(self):
        if self._mean is None:
            flat = rearrange(self.data, "b n d -> (b n) d")
            self._mean = torch.mean(flat, dim=0)
        return self._mean

    @mean.setter
    def mean(self, mean):
        self._mean = mean

    @property
    def var(self):
        if self._var is None:
            flat = rearrange(self.data, "b n d -> (b n) d")
            self._var = torch.var(flat, dim=0, unbiased=False)
        return self._var

    @var.setter
    def var(self, var):
        self._var = var


def load_embeddings(paths_inputs, paths_embeds, idx_layer=2, hidden_dim=128):
    # Prepare dictionary for each category
    embeddings = {
        category: {key: EmbeddingData() for key in mapping.values()}
        for category, mapping in label_map.items()
    }

    for path_input, path_embs in tqdm(
        zip(paths_inputs, paths_embeds), total=len(paths_inputs)
    ):
        inputs = torch.load(path_input, map_location=torch.device("cpu"))
        embeds = torch.load(path_embs, map_location=torch.device("cpu"))

        # For each category, read the labels, map them, and append the corresponding embeddings
        for category in ["speed", "acceleration", "direction", "agent"]:
            for idx_batch, label in enumerate(inputs[f"{category}_labels"]):
                if label.item() in label_map[category].keys():
                    cat_key = label_map[category][label.item()]
                    embeddings[category][cat_key].data.append(
                        embeds[idx_layer][idx_batch]
                    )

    for category in embeddings.keys():
        for cat_key in embeddings[category]:
            embeddings[category][cat_key].data = torch.stack(
                embeddings[category][cat_key].data, dim=0
            )

    return embeddings
