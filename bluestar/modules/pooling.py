import torch
import torch.nn as nn

__all__ = ["GlobalAvgPool2d"]


class GlobalAvgPool2d(nn.Module):

    def __init__(self, keepdim: bool = False):
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Global Average Pooling for tensor
        :param x:       (..., height, width)
        :return:        (...,) or (..., 1, 1)
        """
        y = torch.mean(x, dim=[-2, -1], keepdim=self.keepdim)
        return y
