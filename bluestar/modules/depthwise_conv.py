from typing import Tuple, Union

from .convbn import ConvBN2d, DTYPE_INT2

__all__ = ["DepthwiseConvBN2d"]


class DepthwiseConvBN2d(ConvBN2d):
    """Conv2d + BatchNorm, restricted to depthwise convolution."""

    def __init__(
            self,
            in_channels: int,
            kernel_size: DTYPE_INT2,
            stride: DTYPE_INT2,
            padding: Union[int, str, Tuple[int, int]] = 0,
            dilation: DTYPE_INT2 = 1,
            *, padding_mode: str = "zeros",
            momentum: float = 0.1,
            eps: float = 1e-5,
    ) -> None:
        super().__init__(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, padding_mode=padding_mode,
            momentum=momentum, eps=eps,
        )
