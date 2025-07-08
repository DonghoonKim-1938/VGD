import torch
import torch.nn as nn

from .pooling import GlobalAvgPool2d

__all__ = ["SqueezeExcite"]


class SqueezeExcite(nn.Module):

    def __init__(
            self,
            ch: int,
            reduce_ch: int,
            *, activation_layer=nn.ReLU,
    ) -> None:
        super().__init__()

        self.pool = GlobalAvgPool2d(keepdim=True)
        self.fc1 = nn.Conv2d(ch, reduce_ch, 1, stride=1)
        self.act = activation_layer()
        self.fc2 = nn.Conv2d(reduce_ch, ch, 1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SE layer forward.
        :param x:   (batch_size, channels, h, w)
        :return:    (batch_size, channels, h, w)
        """
        identity = x

        x = self.pool(x)  # (b, ch, 1, 1)
        x = self.fc1(x)  # (b, reduced_ch, 1, 1)
        x = self.act(x)
        x = self.fc2(x)  # (b, ch, 1, 1)
        x = self.sigmoid(x)

        x = identity * x
        return x
