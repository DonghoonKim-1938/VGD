import copy
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .convbn import ConvBN2d, DTYPE_INT2  # noqa

__all__ = ["RepVGGConvBN2d"]


class RepVGGConvBN2d(nn.Module):
    """RepVGG: Making VGG-style ConvNets Great Again"""

    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            kernel_size: DTYPE_INT2,
            stride: DTYPE_INT2 = 1,
            padding: DTYPE_INT2 = 0,
            dilation: DTYPE_INT2 = 1,
            *, groups: int = 1,
            padding_mode: str = "zeros",
            momentum: float = 0.1,
            eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"RepVGG conv kernel should be odd, got {kernel_size}.")

        self.conv_kxk = ConvBN2d(
            input_ch, output_ch, kernel_size, stride, padding, dilation,
            groups=groups, padding_mode=padding_mode, momentum=momentum, eps=eps
        )
        self.conv_1x1 = ConvBN2d(
            input_ch, output_ch, 1, stride, padding - kernel_size // 2,
            groups=groups, momentum=momentum, eps=eps
        )

        if (input_ch == output_ch) and (stride == 1):
            self.bn = nn.BatchNorm2d(input_ch, momentum=momentum, eps=eps)
        else:
            self.bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RepVGG Conv2d forward function.
        :param x:       (batch_size, input_ch, input_h, input_w)
        :return:        (batch_size, output_ch, output_h, output_w)
        """
        x_kxk = self.conv_kxk(x)
        x_1x1 = self.conv_1x1(x)

        y = x_kxk + x_1x1

        if self.bn is not None:
            y = y + self.bn(x)
        return y

    def fuse(self) -> nn.Conv2d:
        self.eval()
        out = copy.deepcopy(self.conv_kxk.conv)
        out.bias = nn.Parameter(torch.zeros(out.out_channels))  # previously bias was None

        # fuse modules
        fused_conv_kxk = self.conv_kxk.fuse()
        fused_conv_1x1 = self.conv_1x1.fuse()

        k = self.conv_kxk.conv.kernel_size[0]  # int
        p = k // 2
        out_w = fused_conv_kxk.weight.data
        out_w += F.pad(fused_conv_1x1.weight.data, [p, p, p, p])

        out_b = fused_conv_kxk.bias.data
        out_b += fused_conv_1x1.bias.data

        if self.bn is not None:
            # fuse bn
            bn_inv_std = torch.rsqrt(self.bn.running_var + self.bn.eps)
            fused_bn_w = self.bn.weight * bn_inv_std
            fused_bn_b = self.bn.bias - (self.bn.running_mean * fused_bn_w)

            fused_identity_w = torch.zeros_like(out_w)
            for i in range(self.conv_kxk.conv.in_channels):
                fused_identity_w[i, i, p, p] = fused_bn_w[i]
            out_w += fused_identity_w
            out_b += fused_bn_b

        out.weight.data.copy_(out_w)
        out.bias.data.copy_(out_b)
        return out.eval()

    @classmethod
    def fuse_repvgg_(cls, base: nn.Module) -> nn.Module:
        base.eval()
        for child_name, child in base.named_children():
            if isinstance(child, RepVGGConvBN2d):
                new_child = child.fuse()
                setattr(base, child_name, new_child)
            else:
                _ = cls.fuse_repvgg_(child)
        return base
