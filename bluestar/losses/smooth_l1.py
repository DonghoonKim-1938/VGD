import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from bluestar.losses.base_loss import LossBase, register_loss

__all__ = ["SmoothL1Loss"]


@register_loss("smooth_l1")
class SmoothL1Loss(LossBase):

    def __init__(
            self,
            beta: float = 1.0,
            reduction: str = "sum",  # default is 'sum' for detection
    ) -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction

        self.loss_fn = nn.SmoothL1Loss(beta=beta, reduction=reduction)

    def forward(
            self,
            deltas: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Smooth L1 loss.
        :param deltas:      (batch_size, ...,)
        :param target:      (batch_size, ...,)
        :return:            scalar              if reduction == 'sum' or 'mean'
                            (batch_size, ...)   if reduction == 'none'
        """
        loss = self.loss_fn(deltas, target)
        return loss
