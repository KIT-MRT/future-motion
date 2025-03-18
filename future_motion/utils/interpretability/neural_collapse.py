import torch

from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear

from typing import Dict, Tuple, Sequence
from pytorch_lightning import LightningModule


class OnlineLinearClassifier(LightningModule):
    def __init__(
        self,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
        log_prefix: str = "direction",
    ) -> None:
        """Adapted from https://github.com/lightly-ai/lightly"""
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.topk = topk
        self.log_prefix = log_prefix

        self.classification_head = Linear(feature_dim, num_classes)
        self.criterion = CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.classification_head(x.detach().flatten(start_dim=1)) # flatten needed? -> doesn't change (8, 256) -> (8, 256)

    def shared_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[int, Tensor]]:
        features, targets = batch[0], batch[1]
        predictions = self.forward(features)
        loss = self.criterion(predictions, targets)
        _, predicted_classes = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_classes, targets, k=self.topk)
        return loss, topk

    def training_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {f"train_{self.log_prefix}_online_cls_loss": loss}
        log_dict.update({f"train_{self.log_prefix}_online_cls_top{k}": acc for k, acc in topk.items()})
        return loss, log_dict

    def validation_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {f"val_{self.log_prefix}_online_cls_loss": loss}
        log_dict.update({f"val_{self.log_prefix}_online_cls_top{k}": acc for k, acc in topk.items()})
        return loss, log_dict
    

def mean_topk_accuracy(
    predicted_classes: Tensor, targets: Tensor, k: Sequence[int]
) -> Dict[int, Tensor]:
    """Computes the mean accuracy for the specified values of k.

    The mean is calculated over the batch dimension.

    Args:
        predicted_classes:
            Tensor of shape (batch_size, num_classes) with the predicted classes sorted
            in descending order of confidence.
        targets:
            Tensor of shape (batch_size) containing the target classes.
        k:
            Sequence of integers specifying the values of k for which the accuracy
            should be computed.

    Returns:
        Dictionary containing the mean accuracy for each value of k. For example for
        k=(1, 5) the dictionary could look like this: {1: 0.4, 5: 0.6}.
    """
    accuracy = {}
    targets = targets.unsqueeze(1)
    with torch.no_grad():
        for num_k in k:
            correct = torch.eq(predicted_classes[:, :num_k], targets)
            accuracy[num_k] = correct.float().sum() / targets.shape[0]
    return accuracy