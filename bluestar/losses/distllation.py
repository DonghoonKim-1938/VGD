import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torch.distributed as dist

from bluestar.losses.base_loss import LossBase, register_loss

__all__ = ["DistilProbLoss", "DistilDINOLoss"]


@register_loss("distil_prob")
class DistilProbLoss(LossBase):

    def __init__(
            self,
            temperature: float = 1.0,
            reduction: str = "batchmean",
            *, scaling: bool = True
    ):
        super().__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.scale = (temperature ** 2) if scaling else 1.0

        self.loss_fn = nn.KLDivLoss(
            reduction=reduction,
            log_target=True  # non-default
        )

    def forward(
            self,
            student_logit: torch.Tensor,  # output of FC before softmax
            teacher_logit: torch.Tensor,  # output of FC before softmax
    ) -> torch.Tensor:
        """Distillation loss for the final label prediction.
        :param student_logit:       (batch_size, num_classes)
        :param teacher_logit:       (batch_size, num_classes)
        :return:                    scalar                      if reduction == 'sum', 'mean', or 'batchmean'
                                    (batch_size, num_classes)   if reduction == 'none'
        """
        student_log_prob = F.log_softmax(student_logit / self.temperature, dim=-1)
        teacher_log_prob = F.log_softmax(teacher_logit / self.temperature, dim=-1)  # target

        loss = self.loss_fn(student_log_prob, teacher_log_prob)
        loss *= self.scale
        return loss


@register_loss("distil_dino")
class DistilDINOLoss(LossBase):
    """DINO loss from:
    Emerging Properties in Self-Supervised Vision Transformers (ICCV 2021)
    """

    def __init__(
            self,
            feature_dim: int,
            student_temperature: float = 0.1,
            teacher_temperature: float = 0.04,
            reduction: str = "batchmean",
            *, center_momentum: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.reduction = reduction
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.center_momentum = center_momentum

        center = torch.zeros(1, feature_dim)
        self.register_buffer("center", center)

    @torch.no_grad()
    def _update_center(self, logit: torch.Tensor) -> None:
        """EMA update of feature means like batch normalization."""
        # logit: (batch_size, feature_dim)
        center_sum = torch.sum(logit, dim=0)  # (feature_dim,)
        batch_size = logit.shape[0]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(center_sum)  # sum through GPUs
            batch_size *= dist.get_world_size()  # assert all GPUs process the same number of samples
        new_center = center_sum / batch_size

        # update (3 line below are all same)
        # self.center = self.center * (1.0 - self.center_momentum) + new_center * self.center_momentum
        # self.center = self.center + self.center_momentum * (new_center - self.center)
        self.center.add_(new_center - self.center, alpha=self.center_momentum)

    def forward(
            self,
            student_logit: torch.Tensor,  # output of FC before softmax
            teacher_logit: torch.Tensor,  # output of FC before softmax
    ) -> torch.Tensor:
        """Distillation loss for DINO

        Note that the original DINO takes multiple views of student and teachers.
        (ex: 5 views obtained from student & 5 views from teacher)
        In that case, every pair of student-teacher views should be compared,
        unless two views correspond to the same region of the image.
        This implementation only compares 1-view and 1-view of teacher and student, respectively.

        :param student_logit:     (batch_size, feature_dim)
        :param teacher_logit:     (batch_size, feature_dim)
        :return:                  scalar                      if reduction == 'sum', 'mean', or 'batchmean'
                                  (batch_size, feature_dim)   if reduction == 'none'
        """
        # -------- student -------- #
        student_logit = F.log_softmax(student_logit / self.student_temperature, dim=-1)

        # -------- teacher -------- #
        teacher_logit = teacher_logit.detach()
        teacher_prob = F.softmax((teacher_logit - self.center) / self.teacher_temperature, dim=-1)
        if self.training:
            self._update_center(teacher_logit)

        # -------- loss computation -------- #
        loss = -(teacher_prob * student_logit)  # (b, c), per-element cross-entropy
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "batchmean":  # mathematically correct cross-entropy
            return loss.sum(dim=-1).mean()  # (b, c) -> (b,) -> scalar
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction type {self.reduction}.")
