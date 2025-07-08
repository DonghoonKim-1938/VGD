import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .convbn import DTYPE_INT2  # noqa

__all__ = ["StdConv2d"]


class StdConv2d(nn.Conv2d):

    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            kernel_size: DTYPE_INT2,
            stride: DTYPE_INT2 = 1,
            padding: DTYPE_INT2 = 0,
            dilation: DTYPE_INT2 = 1,
            *, groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros"
    ) -> None:
        super().__init__(input_ch, output_ch, kernel_size, stride, padding, dilation,
                         groups, bias=bias, padding_mode=padding_mode)
        self._fused = False

    def _standardize_weight(self) -> torch.Tensor:
        """Standardize conv weight"""
        w = self.weight
        ch_var, ch_mean = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        ch_inv_sqrt = torch.rsqrt(ch_var + 1e-5)
        w = (w - ch_mean) * ch_inv_sqrt
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused:
            return super().forward(x)

        w = self._standardize_weight()
        y = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y

    @torch.no_grad()
    def fuse(self) -> None:
        """Fuse weight, cannot roll back.
        Unlike ConvBN, this `fuse' does not return another modules.
        """
        w = self._standardize_weight()
        self.weight.data.copy_(w)
        self._fused = True
