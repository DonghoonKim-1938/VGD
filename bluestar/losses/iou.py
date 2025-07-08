import torch

from bluestar.losses.base_loss import LossBase, register_loss

__all__ = ["IoULoss"]


@register_loss("iou")
class IoULoss(LossBase):
    """Wrapper of GIoU and DIoU loss.
    CIoU loss not yet implemented.
    """

    def __init__(
            self,
            loss_type: str = "giou",
            eps: float = 1e-6,
            reduction: str = "sum",  # default is 'sum' for detection
    ) -> None:
        super().__init__()
        loss_type = loss_type.lower()
        if loss_type not in ("giou", "diou"):
            raise ValueError(f"IoU loss type only supports giou and diou, got {loss_type}.")
        self.loss_type = loss_type
        self.eps = eps
        self.reduction = reduction

    def forward(
            self,
            pred_boxes: torch.Tensor,  # xyxy format
            target_boxes: torch.Tensor,  # xyxy format
    ) -> torch.Tensor:
        """IoU Loss.
        :param pred_boxes:      (batch_size, 4)
        :param target_boxes:    (batch_size, 4)
        :return:            scalar          if reduction == 'sum' or 'mean'
                            (batch_size,)   if reduction == 'none'
        :note:
            make sure that box format for input arguments is (x1, y1, x2, y2).
        """
        if (target_boxes.numel() == 0) or (pred_boxes.numel() == 0):  # shortcut
            return torch.tensor(0.0, dtype=torch.float32, device=pred_boxes.device)

        if self.loss_type == "giou":
            return self._forward_giou(pred_boxes, target_boxes)
        elif self.loss_type == "diou":
            return self._forward_diou(pred_boxes, target_boxes)
        else:
            raise NotImplementedError

    def _forward_giou(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = pred_boxes.unbind(dim=-1)
        tx1, ty1, tx2, ty2 = target_boxes.unbind(dim=-1)

        # iou
        inter_x1 = torch.max(x1, tx1)
        inter_y1 = torch.max(y1, ty1)
        inter_x2 = torch.min(x2, tx2)
        inter_y2 = torch.min(y2, ty2)

        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        intersection = torch.clamp_min(intersection, self.eps)  # remove invalids
        union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1)
        union = union - intersection
        iou = intersection / (union + self.eps)

        # smallest enclosure
        enc_x1 = torch.min(x1, tx1)
        enc_y1 = torch.min(y1, ty1)
        enc_x2 = torch.max(x2, tx2)
        enc_y2 = torch.max(y2, ty2)

        enclosure = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        remains = (enclosure - union) / (enclosure + self.eps)

        # GIoU loss: increase iou, decrease remains.
        loss = 1.0 - (iou - remains)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss

    def _forward_diou(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = pred_boxes.unbind(dim=-1)
        tx1, ty1, tx2, ty2 = target_boxes.unbind(dim=-1)

        # iou
        inter_x1 = torch.max(x1, tx1)
        inter_y1 = torch.max(y1, ty1)
        inter_x2 = torch.min(x2, tx2)
        inter_y2 = torch.min(y2, ty2)

        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        intersection = torch.clamp_min(intersection, self.eps)  # remove invalids
        union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1)
        union = union - intersection
        iou = intersection / (union + self.eps)

        # smallest enclosure's diagonal
        enc_x1 = torch.min(x1, tx1)
        enc_y1 = torch.min(y1, ty1)
        enc_x2 = torch.max(x2, tx2)
        enc_y2 = torch.max(y2, ty2)

        diagonal = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

        # center distance
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        tcx = (tx1 + tx2) * 0.5
        tcy = (ty1 + ty2) * 0.5

        distance = (cx - tcx) ** 2 + (cy - tcy) ** 2

        # DIoU loss: increase iou, decrease distance, increase diagonal.
        loss = 1.0 - (iou - distance / (diagonal + self.eps))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss
