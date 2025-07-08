import torch
import torch.nn as nn

from .base_head import VisionHeadBase, register_vision_head

__all__ = ["ClassificationHead"]


@register_vision_head("cls")
class ClassificationHead(VisionHeadBase):

    def __init__(
            self,
            feature_dim: int,
            num_classes: int,
            drop_prob: float = 0.0,
            bias: bool = True
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.drop = nn.Dropout(p=drop_prob, inplace=True)
        self.linear = nn.Linear(feature_dim, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classification head forward
        :param x:   (batch_size, feature_dim)
        :return:    (batch_size, num_classes)
        """
        x = self.drop(x)
        logit = self.linear(x)
        return logit

    def forward_prob(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.forward(x)
        prob = torch.softmax(logit, dim=1)
        return prob

    def forward_pred(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.forward(x)
        prediction = torch.argmax(logit, dim=1, keepdim=False)  # (batch_size,)
        return prediction
