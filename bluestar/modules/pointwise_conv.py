from .convbn import ConvBN2d

__all__ = ["PointwiseConvBN2d"]


class PointwiseConvBN2d(ConvBN2d):
    """Conv2d + BatchNorm, restricted to pointwise convolution."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *, groups: int = 1,
            padding_mode: str = "zeros",
            momentum: float = 0.1,
            eps: float = 1e-5,
    ) -> None:
        super().__init__(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0,
            groups=groups, padding_mode=padding_mode,
            momentum=momentum, eps=eps,
        )
