from typing import Tuple, Union
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from bluestar.modules.batchnorm import MyBatchNorm2d

__all__ = ["ConvBN2d", "DTYPE_INT2"]

DTYPE_INT2 = Union[int, Tuple[int, int]]


class ConvBN2d(nn.Module):
    """Conv2d + BatchNorm"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: DTYPE_INT2,
            stride: DTYPE_INT2 = 1,
            padding: Union[int, str, Tuple[int, int]] = 0,
            dilation: DTYPE_INT2 = 1,
            *, groups: int = 1,
            padding_mode: str = "zeros",
            momentum: float = 0.1,
            eps: float = 1e-5,
            track_running_stats=True,
            manual_batchnorm=False
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False, padding_mode=padding_mode
        )

        self.bn = nn.BatchNorm2d(
            out_channels, eps=eps, momentum=momentum, track_running_stats=track_running_stats
        ) if manual_batchnorm is False \
            else MyBatchNorm2d(out_channels, eps=eps, momentum=momentum, track_running_stats=track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv2d + BatchNorm2d forward function.
        :param x:       (batch_size, input_ch, input_h, input_w)
        :return:        (batch_size, output_ch, output_h, output_w)
        """
        x = self.conv(x)
        x = self.bn(x)
        return x

    def fuse(self) -> nn.Conv2d:
        self.eval()
        fused_conv = fuse_conv_bn_eval(self.conv, self.bn)  # , transpose=False)
        return fused_conv.eval()

    @classmethod
    def fuse_conv_bn_(cls, base: nn.Module) -> nn.Module:
        base.eval()
        for child_name, child in base.named_children():
            if isinstance(child, ConvBN2d):
                new_child = child.fuse()
                setattr(base, child_name, new_child)
            else:
                _ = cls.fuse_conv_bn_(child)
        return base
