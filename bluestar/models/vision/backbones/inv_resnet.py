"""
Implementation for "Invertible Residual Network" (ICML, 2019)
http://proceedings.mlr.press/v97/behrmann19a.html
by DH Kim
"""

from typing import List
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from bluestar.modules import ActNorm2d
from .base_backbone import VisionBackboneBase, register_vision_backbone

__all__ = [
    # inv-resnet blocks
    "InvResNetBottleneckBlock",
    # inv-resnet back bones
    "InvResNet39Backbone",
]


class InvResNetBottleneckBlock(nn.Module):

    def __init__(
            self,
            input_ch: int,
            middle_ch: int,
            output_ch: int,
            stride: int = 1,
            *, groups: int = 1,
    ) -> None:
        super().__init__()
        assert stride in (1, 2)

        self.input_ch = input_ch
        self.middle_ch = middle_ch
        self.output_ch = output_ch
        self.stride = stride  # For squeezing the spatial info to channel info
        self.groups = groups

        # squeezing the spatial information to channel
        if stride == 2:
            assert output_ch == input_ch * 4
            self.squeeze = nn.PixelUnshuffle(downscale_factor=2)
            self.unsqueeze = nn.PixelShuffle(upscale_factor=2)
        else:
            assert output_ch == input_ch
            self.squeeze = self.unsqueeze = None

        self.act_in = ActNorm2d(num_channel=output_ch)

        self.conv1 = spectral_norm(nn.Conv2d(output_ch, middle_ch, kernel_size=3, stride=1, padding=1))
        self.act1 = nn.ELU(inplace=True)
        self.conv2 = spectral_norm(nn.Conv2d(middle_ch, middle_ch, kernel_size=1, stride=1, padding=0))
        self.act2 = nn.ELU(inplace=True)
        self.conv3 = spectral_norm(nn.Conv2d(middle_ch, output_ch, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Invertible ResNetBottleneckBlock forward
        :param x:   (batch_size, input_ch, h, w)
        :return:    (batch_size, output_ch, h/s, w/s)
        """

        if self.squeeze is not None:
            x = self.squeeze(x)

        identity = x
        x = self.act_in(x)  # pre-norm
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)

        x = x + identity
        return x

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, max_iter: int = 100) -> torch.Tensor:
        """Inverse operation of ResNet block: fixed-point iteration
        :param z:           (batch_size, output_ch, h, w)
        :param max_iter:    number of iterations
        :return:            (batch_size, output_ch/(s^2), h*s, w*s)
        """
        x = z  # init
        for i in range(max_iter):
            x_sum = self.conv1(x)
            x_sum = self.act1(x_sum)
            x_sum = self.conv2(x_sum)
            x_sum = self.act2(x_sum)
            x_sum = self.conv3(x_sum)
            x = z - x_sum

        x = self.act_in.inverse(x)

        if self.unsqueeze is not None:
            x = self.unsqueeze(x)
        return x


class InvResNetWithBottleneckBlock(VisionBackboneBase):

    def __init__(self, num_layers: List[int]):
        super().__init__()
        if len(num_layers) != 3:
            raise ValueError(f"[ERROR] Invertible ResNet needs three #layers, got {num_layers}.")
        self.num_layers = num_layers

        self.init_squeeze = nn.PixelUnshuffle(downscale_factor=2)  # 3 -> 12
        self.init_unsqueeze = nn.PixelShuffle(upscale_factor=2)  # 12 -> 3
        self._features_ch.append(12)

        stage1 = []
        for i in range(num_layers[0]):
            stage1.append(InvResNetBottleneckBlock(12, 32, 12, stride=1))
        self.stage1 = nn.ModuleList(stage1)
        self._features_ch.append(12)

        stage2 = []
        for i in range(num_layers[1]):
            if i == 0:
                stage2.append(InvResNetBottleneckBlock(12, 32, 48, stride=2))
            else:
                stage2.append(InvResNetBottleneckBlock(48, 32, 48, stride=1))
        self.stage2 = nn.ModuleList(stage2)
        self._features_ch.append(48)

        stage3 = []
        for i in range(num_layers[2]):
            if i == 0:
                stage3.append(InvResNetBottleneckBlock(48, 32, 192, stride=2))
            else:
                stage3.append(InvResNetBottleneckBlock(192, 32, 192, stride=1))
        self.stage3 = nn.ModuleList(stage3)
        self._features_ch.append(192)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.init_squeeze(x)  # (3, 32, 32) -> (12, 16, 16)
        features.append(x)

        for block in self.stage1:  # (12, 16, 16) -> (12, 16, 16)
            x = block(x)
        features.append(x)

        for block in self.stage2:  # (12, 16, 16) -> (48, 8, 8)
            x = block(x)
        features.append(x)

        for block in self.stage3:  # (48, 8, 8) -> (192, 4, 4)
            x = block(x)
        features.append(x)

        return features

    @torch.no_grad()
    def inverse(self, z, max_iter=10):
        inv_features = []

        for i in range(len(self.stage3)):  # (192, 4, 4) -> (48, 8, 8)
            z = self.stage3[-1 - i].inverse(z, max_iter=max_iter)
        inv_features.append(z)

        for i in range(len(self.stage2)):  # (48, 8, 8) -> (12, 16, 16)
            z = self.stage2[-1 - i].inverse(z, max_iter=max_iter)
        inv_features.append(z)

        for i in range(len(self.stage1)):  # (12, 16, 16) -> (12, 16, 16)
            z = self.stage1[-1 - i].inverse(z, max_iter=max_iter)
        inv_features.append(z)

        z = self.init_unsqueeze(z)
        inv_features.append(z)

        return inv_features


@register_vision_backbone("inv_resnet39")
class InvResNet39Backbone(InvResNetWithBottleneckBlock):

    def __init__(self):
        super().__init__(num_layers=[4, 4, 4])
