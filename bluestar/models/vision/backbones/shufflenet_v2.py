from typing import List
import torch
import torch.nn as nn

from bluestar.modules.convbn import ConvBN2d
from bluestar.modules.pointwise_conv import PointwiseConvBN2d
from bluestar.modules.depthwise_conv import DepthwiseConvBN2d
from .base_backbone import VisionBackboneBase, register_vision_backbone

__all__ = [
    "ShuffleNetV2_X0_5", "ShuffleNetV2_X1_0", "ShuffleNetV2_X1_5", "ShuffleNetV2_X2_0"
]


class ShuffleNetV2Block(nn.Module):
    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            stride: int = 1,
    ) -> None:
        super().__init__()

        self.stride = stride
        assert stride in (1, 2)

        split_ch = output_ch // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                DepthwiseConvBN2d(input_ch, kernel_size=3, stride=self.stride, padding=1),
                PointwiseConvBN2d(input_ch, split_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = None

        self.branch2 = nn.Sequential(
            PointwiseConvBN2d(input_ch if (self.stride > 1) else split_ch, split_ch),
            nn.ReLU(inplace=True),
            DepthwiseConvBN2d(split_ch, kernel_size=3, stride=self.stride, padding=1),
            PointwiseConvBN2d(split_ch, split_ch),
            nn.ReLU(inplace=True)
        )
        self.shuffle = nn.ChannelShuffle(groups=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # input shape : (B, C, H, W)
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = self.shuffle(out)
        return out


class ShuffleNetV2Backbone(VisionBackboneBase):
    def __init__(
            self,
            num_layers: List[int],
            channels: List[int],  # [24, 48, 96, 192, 1024]
    ) -> None:
        super().__init__()

        if len(num_layers) != 3:
            raise ValueError(f"[ERROR] ShuffleNetV2 needs three #num_layers, got {num_layers}.")
        if len(channels) != 5:
            raise ValueError(f"[ERROR] ShuffleNetV2 needs five #channels, got {channels}.")
        self.num_layers = num_layers
        self.channels = channels

        input_ch = 3
        output_ch = self.channels[0]

        self.stem = nn.Sequential(
            ConvBN2d(input_ch, output_ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (224, 224) -> (112, 112) -> (56, 56)
        )  # (224, 224) -> (112, 112)
        self._features_ch.append(output_ch)

        input_ch = output_ch
        output_ch = channels[1]
        stage1 = []
        for i in range(num_layers[0]):
            if i == 0:
                stage1.append(ShuffleNetV2Block(input_ch, output_ch, stride=2))
            else:
                stage1.append(ShuffleNetV2Block(output_ch, output_ch, stride=1))
        self.stage1 = nn.ModuleList(stage1)
        self._features_ch.append(output_ch)

        input_ch = output_ch
        output_ch = channels[2]
        stage2 = []
        for i in range(num_layers[1]):
            if i == 0:
                stage2.append(ShuffleNetV2Block(input_ch, output_ch, stride=2))
            else:
                stage2.append(ShuffleNetV2Block(output_ch, output_ch, stride=1))
        self.stage2 = nn.ModuleList(stage2)
        self._features_ch.append(output_ch)

        input_ch = output_ch
        output_ch = channels[3]
        stage3 = []
        for i in range(num_layers[2]):
            if i == 0:
                stage3.append(ShuffleNetV2Block(input_ch, output_ch, stride=2))
            else:
                stage3.append(ShuffleNetV2Block(output_ch, output_ch, stride=1))
        self.stage3 = nn.ModuleList(stage3)
        self._features_ch.append(output_ch)

        input_ch = output_ch
        output_ch = self.channels[-1]  # 4

        self.last_conv = nn.Sequential(
            ConvBN2d(input_ch, output_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self._features_ch.append(output_ch)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.stem(x)  # (3, 224, 224) -> (24, 56, 56)
        features.append(x)

        for block in self.stage1:  # (24, 56, 56) -> (116, 28, 28)
            x = block(x)
        features.append(x)

        for block in self.stage2:  # (116, 28, 28) -> (232, 14, 14)
            x = block(x)
        features.append(x)

        for block in self.stage3:  # (232, 14, 14) -> (464, 7, 7)
            x = block(x)
        features.append(x)

        x = self.last_conv(x)  # (464, 7, 7) -> (1024, 7, 7)
        features.append(x)

        return features


@register_vision_backbone("shufflenetv2_x0_5")
class ShuffleNetV2_X0_5(ShuffleNetV2Backbone):

    def __init__(self):
        super().__init__(num_layers=[4, 8, 4], channels=[24, 48, 96, 192, 1024])


@register_vision_backbone("shufflenetv2_x1_0")
class ShuffleNetV2_X1_0(ShuffleNetV2Backbone):

    def __init__(self):
        super().__init__(num_layers=[4, 8, 4], channels=[24, 116, 232, 464, 1024])


@register_vision_backbone("shufflenetv2_x1_5")
class ShuffleNetV2_X1_5(ShuffleNetV2Backbone):

    def __init__(self):
        super().__init__(num_layers=[4, 8, 4], channels=[24, 176, 352, 704, 1024])


@register_vision_backbone("shufflenetv2_x2_0")
class ShuffleNetV2_X2_0(ShuffleNetV2Backbone):

    def __init__(self):
        super().__init__(num_layers=[4, 8, 4], channels=[24, 244, 488, 976, 2048])
