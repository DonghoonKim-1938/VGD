from typing import Optional, Tuple
import torch
import torch.nn as nn

from .convbn import DTYPE_INT2  # noqa

__all__ = ["OctaveConv2d", "FirstOctaveConv2d", "LastOctaveConv2d"]


class OctaveConv2d(nn.Module):

    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            kernel_size: DTYPE_INT2,
            alpha_in: float,
            alpha_out: Optional[float] = None,
            stride: DTYPE_INT2 = 1,
            padding: DTYPE_INT2 = 0,
            dilation: DTYPE_INT2 = 1,
            *, groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        # ---- condition check ---- #
        if not 0 < alpha_in < 1:
            raise ValueError(f"OctaveConv require alpha_in to be [0, 1], got {alpha_in}.")
        if alpha_out is None:
            alpha_out = alpha_in
        if not 0 < alpha_out < 1:
            raise ValueError(f"OctaveConv require alpha_out to be [0, 1], got {alpha_out}.")
        if stride not in (1, 2):
            raise ValueError(f"OctaveConv stride should be 1 or 2, got {stride}.")

        low_in_ch = min(max(int(input_ch * alpha_in), 1), input_ch - 1)
        low_out_ch = min(max(int(output_ch * alpha_out), 1), output_ch - 1)
        high_in_ch = input_ch - low_in_ch
        high_out_ch = output_ch - low_out_ch

        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.low_in_channels = low_in_ch
        self.low_out_channels = low_out_ch
        self.high_in_channels = high_in_ch
        self.high_out_channels = high_out_ch

        # ---- conv layers ---- #
        conv_kwargs = dict(kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation,
                           groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv_l2l = nn.Conv2d(low_in_ch, low_out_ch, **conv_kwargs)
        self.conv_l2h = nn.Conv2d(low_in_ch, high_out_ch, **conv_kwargs)
        self.conv_h2l = nn.Conv2d(high_in_ch, low_out_ch, **conv_kwargs)
        self.conv_h2h = nn.Conv2d(high_in_ch, high_out_ch, **conv_kwargs)

        # ---- scaling layers ---- #
        self.stride = stride
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """OctaveConv forward function.
        :param x_high:      (batch_size, high_input_ch, input_h, input_w)
        :param x_low:       (batch_size, low_input_ch, input_h / 2, input_w / 2)
        :return:            (batch_size, high_output_ch, output_h, output_w)
                            (batch_size, low_output_ch, output_h / 2, output_w / 2)
        """
        _, high_ch, high_h, high_w = x_high.shape
        _, low_ch, low_h, low_w = x_low.shape
        if (high_ch != self.high_in_channels) or (low_ch != self.low_in_channels):
            raise ValueError(f"Octave conv input shape {x_low.shape} and {x_high.shape} mismatch to"
                             f"input_channels {self.low_in_channels} and {self.high_in_channels}, respectively.")
        if (high_h != low_h * 2) or (high_w != low_w * 2):
            raise ValueError(f"Octave conv high part should have double resolution to low part.")

        if self.stride == 2:
            x_high = self.downsample(x_high)  # (h, w) -> (h/2, w/2)
            x_low = self.downsample(x_low)  # (h/2, w/2) -> (h/4, w/4)

        # comments are for stride==1 case.

        x_l2l = self.conv_l2l(x_low)  # (h/2, w/2)
        x_l2h = self.conv_l2h(x_low)  # (h/2, w/2)
        x_l2h = self.upsample(x_l2h)  # (h, w)

        x_h2l = self.conv_h2l(x_high)  # (h, w)
        x_h2h = self.conv_h2h(x_high)  # (h, w)
        x_h2l = self.downsample(x_h2l)  # (h/2, w/2)

        x_low = x_l2l + x_h2l
        x_high = x_l2h + x_h2h

        return x_high, x_low


class FirstOctaveConv2d(nn.Module):

    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            kernel_size: DTYPE_INT2,
            alpha_out: float,
            stride: DTYPE_INT2 = 1,
            padding: DTYPE_INT2 = 0,
            dilation: DTYPE_INT2 = 1,
            *, groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        # ---- condition check ---- #
        if not 0 < alpha_out < 1:
            raise ValueError(f"OctaveConv require alpha_out to be [0, 1], got {alpha_out}.")
        if stride not in (1, 2):
            raise ValueError(f"OctaveConv stride should be 1 or 2, got {stride}.")

        low_out_ch = min(max(int(output_ch * alpha_out), 1), output_ch - 1)
        high_out_ch = output_ch - low_out_ch

        self.alpha_out = alpha_out
        self.low_out_channels = low_out_ch
        self.high_out_channels = high_out_ch

        # ---- conv layers ---- #
        conv_kwargs = dict(kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation,
                           groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv_h2l = nn.Conv2d(input_ch, low_out_ch, **conv_kwargs)
        self.conv_h2h = nn.Conv2d(input_ch, high_out_ch, **conv_kwargs)

        # ---- scaling layers ---- #
        self.stride = stride
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """OctaveConv forward function (high-only input)
        :param x:           (batch_size, input_ch, input_h, input_w)
        :return:            (batch_size, high_output_ch, output_h, output_w)
                            (batch_size, low_output_ch, output_h / 2, output_w / 2)
        """
        if self.stride == 2:
            x = self.downsample(x)  # (h, w) -> (h/2, w/2)

        # comments are for stride==1 case.

        x_low = self.conv_h2l(x)  # (h, w)
        x_high = self.conv_h2h(x)  # (h, w)
        x_low = self.downsample(x_low)  # (h/2, w/2)

        return x_high, x_low


class LastOctaveConv2d(nn.Module):

    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            kernel_size: DTYPE_INT2,
            alpha_in: float,
            stride: DTYPE_INT2 = 1,
            padding: DTYPE_INT2 = 0,
            dilation: DTYPE_INT2 = 1,
            *, groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        # ---- condition check ---- #
        if not 0 < alpha_in < 1:
            raise ValueError(f"OctaveConv require alpha_in to be [0, 1], got {alpha_in}.")
        if stride not in (1, 2):
            raise ValueError(f"OctaveConv stride should be 1 or 2, got {stride}.")

        low_in_ch = min(max(int(input_ch * alpha_in), 1), input_ch - 1)
        high_in_ch = input_ch - low_in_ch

        self.alpha_in = alpha_in
        self.low_in_channels = low_in_ch
        self.high_in_channels = high_in_ch

        # ---- conv layers ---- #
        conv_kwargs = dict(kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation,
                           groups=groups, bias=bias, padding_mode=padding_mode)
        self.conv_l2h = nn.Conv2d(low_in_ch, output_ch, **conv_kwargs)
        self.conv_h2h = nn.Conv2d(high_in_ch, output_ch, **conv_kwargs)

        # ---- scaling layers ---- #
        self.stride = stride
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """OctaveConv forward function (high-only output).
        :param x_high:      (batch_size, high_input_ch, input_h, input_w)
        :param x_low:       (batch_size, low_input_ch, input_h / 2, input_w / 2)
        :return:            (batch_size, high_output_ch, output_h, output_w)
        """
        _, high_ch, high_h, high_w = x_high.shape
        _, low_ch, low_h, low_w = x_low.shape
        if (high_ch != self.high_in_channels) or (low_ch != self.low_in_channels):
            raise ValueError(f"Octave conv input shape {x_low.shape} and {x_high.shape} mismatch to"
                             f"input_channels {self.low_in_channels} and {self.high_in_channels}, respectively.")
        if (high_h != low_h * 2) or (high_w != low_w * 2):
            raise ValueError(f"Octave conv high part should have double resolution to low part.")

        if self.stride == 2:
            x_high = self.downsample(x_high)  # (h, w) -> (h/2, w/2)
            x_low = self.downsample(x_low)  # (h/2, w/2) -> (h/4, w/4)

        # comments are for stride==1 case.

        x_l2h = self.conv_l2h(x_low)  # (h/2, w/2)
        x_l2h = self.upsample(x_l2h)  # (h, w)

        x_h2h = self.conv_h2h(x_high)  # (h, w)

        x_high = x_l2h + x_h2h

        return x_high
