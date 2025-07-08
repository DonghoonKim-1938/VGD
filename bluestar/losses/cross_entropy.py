import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from bluestar.losses.base_loss import LossBase, register_loss

__all__ = ["CrossEntropyLoss"]


@register_loss("cross_entropy")
class CrossEntropyLoss(LossBase):

    def __init__(
            self,
            label_smoothing: float = 0.0,
            reduction: str = "mean",
            *, ignore_index: int = -1
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def forward(
            self,
            logit: torch.Tensor,  # output of FC before softmax
            target: torch.Tensor,  # integer label
    ) -> torch.Tensor:
        """Cross-entropy loss.
        :param logit:       (batch_size, num_classes)
        :param target:      (batch_size,)
        :return:            scalar          if reduction == 'sum', 'mean' or 'batchmean'
                            (batch_size,)   if reduction == 'none'
        """
        loss = self.loss_fn(logit, target)
        return loss
