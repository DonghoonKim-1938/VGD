import torch
from torchvision.ops.focal_loss import sigmoid_focal_loss

from bluestar.losses.base_loss import LossBase, register_loss

__all__ = ["SigmoidFocalLoss"]


@register_loss("sigmoid_focal")
class SigmoidFocalLoss(LossBase):

    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            reduction: str = "sum",  # default is 'sum' for detection
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.loss_fn = sigmoid_focal_loss

    def forward(
            self,
            logit: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Sigmoid Focal loss.
        :param logit:       (batch_size, ..., num_classes)
        :param target:      (batch_size, ..., num_classes)
        :return:            scalar                          if reduction == 'sum' or 'mean'
                            (batch_size, ..., num_classes)  if reduction == 'none'
        """
        loss = sigmoid_focal_loss(logit, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        return loss
