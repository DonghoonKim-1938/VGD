import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from bluestar.losses.base_loss import LossBase, register_loss

__all__ = ["BinaryCrossEntropyLoss"]


@register_loss("binary_cross_entropy")
class BinaryCrossEntropyLoss(LossBase):

    def __init__(
            self,
            reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction=reduction,
        )

    def forward(
            self,
            logit: torch.Tensor,   # output of FC before softmax
            target: torch.Tensor,  # integer label
    ) -> torch.Tensor:
        """Binary Cross-entropy loss.
        :param logit:       (batch_size, num_classes)
        :param target:      (batch_size,)
        :return:            scalar          if reduction == 'sum', 'mean' or 'batchmean'
                            (batch_size,)   if reduction == 'none'
        """
        loss = self.loss_fn(logit, target)
        return loss
