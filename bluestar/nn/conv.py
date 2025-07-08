from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Conv2d"]


class Conv2d(nn.Module):
    """Basic convolution layer for 2D image.
    This class is written only for example.
    """

    def __init__(self,
                 input_ch: int,
                 output_ch: int,
                 kernel_size: int,  # this layer only support square kernel
                 stride: int = 1,  # this layer only support square stride
                 padding: int = 0,  # this layer only support same padding for all edges
                 *, dilation: int = 1,  # this layer only support square dilation
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros"):
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        if padding_mode != "zeros":
            raise ValueError("Conv2d example only support zeros for now")

        weight = torch.zeros(output_ch, input_ch // groups, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(weight, a=2.0, mode="fan_out")
        self.weight = nn.Parameter(weight)

        if self.use_bias:
            bias = torch.zeros(output_ch)
            nn.init.zeros_(bias)
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv2d forward function.
        :param x:       (batch_size, output_ch, input_h, input_w)
        :return:        (batch_size, output_ch, output_h, output_w)
        """
        y = F.conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return y

    def extra_repr(self) -> str:
        return f"output_ch={self.input_ch}, output_ch={self.output_ch}, " \
               f"stride={self.stride}, padding={self.padding}, " \
               f"dilation={self.dilation}, groups={self.groups}, " \
               f"bias={self.use_bias}, padding_mode={self.padding_mode}"


if __name__ == '__main__':
    m = Conv2d(16, 8, kernel_size=3, stride=2, padding=1, dilation=1, groups=2, bias=True)
    t_in = torch.empty(6, 16, 32, 32)
    t_out = m(t_in)
    print(t_out.shape)
    print(m)
    print(m.weight.shape)
