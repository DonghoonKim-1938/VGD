import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from bluestar.losses.base_loss import LossBase, register_loss

__all__ = ["KLdivergenceLoss"]


@register_loss("kl_divergence")
class KLdivergenceLoss(LossBase):

    def __init__(
            self,
            reduction: str = "batchmean",
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_fn = nn.KLDivLoss(
            reduction=reduction,
        )

    def forward(
            self,
            logit: torch.Tensor,  # output of FC before softmax
            target: torch.Tensor,  # probability (not log)
    ) -> torch.Tensor:
        """KL-divergence loss.
        :param logit:       (batch_size, num_classes)
        :param target:      (batch_size,)
        :return:            scalar          if reduction == 'sum', 'mean' or 'batchmean'
                            (batch_size,)   if reduction == 'none'
        """
        loss = self.loss_fn(logit, target)
        return loss
