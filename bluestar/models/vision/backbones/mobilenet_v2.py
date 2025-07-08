from typing import List
import torch
import torch.nn as nn

from bluestar.modules.convbn import ConvBN2d
from bluestar.modules.depthwise_conv import DepthwiseConvBN2d
from bluestar.modules.pointwise_conv import PointwiseConvBN2d
from .base_backbone import VisionBackboneBase, register_vision_backbone

__all__ = ["MobileNetV2Backbone"]


class MobileNetV2StemBlock(nn.Module):

    def __init__(self, ch: int = 32):
        super().__init__()
        self.ch = ch
        self.conv = ConvBN2d(3, ch, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MobileNetV2StemBlock forward.
        :param x:       (batch_size, 3, h, w)
        :return:        (batch_size, ch, h/2, w/2)
        """
        x = self.conv(x)
        x = self.act(x)
        return x


class MobileNetV2OutBlock(nn.Module):

    def __init__(self, input_ch: int, output_ch: int = 1280):
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.conv = PointwiseConvBN2d(input_ch, output_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MobileNetV2OutBlock forward
        :param x:   (batch_size, input_ch, h, w)
        :return:    (batch_size, output_ch, h, w)
        """
        x = self.conv(x)
        x = self.act(x)
        return x


class InvertedBottleneckBlock(nn.Module):

    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            ratio: int,
            stride: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        if self.stride != 1 and self.stride != 2:
            raise ValueError(f"[ERROR] MobileNetV2 needs stride 1 or 2 . {stride} is not supported.")
        self.has_skip_connection = (self.stride == 1) and (input_ch == output_ch)

        # channel expansion
        hidden_ch = input_ch * ratio

        layers: List[nn.Module] = []
        if ratio != 1:
            layers.append(PointwiseConvBN2d(input_ch, hidden_ch))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            DepthwiseConvBN2d(hidden_ch, kernel_size=3, stride=stride, padding=1),
            nn.ReLU6(inplace=True),
            PointwiseConvBN2d(hidden_ch, output_ch)
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """InvertedBottleneckBlock forward
        :param x:   (batch_size, input_ch, h, w)
        :return:    (batch_size, output_ch, h/s, w/s)
        """
        if self.has_skip_connection:
            return x + self.layers(x)
        else:
            return self.layers(x)


class _MobileNetV2Base(VisionBackboneBase):

    def __init__(self, num_layers: List[int]):
        super().__init__()
        if len(num_layers) != 7:
            raise ValueError(f"[ERROR] MobileNetV2 needs seven #layers, got {num_layers}.")
        self.num_layers = num_layers

        self.stem = MobileNetV2StemBlock(32)
        self._features_ch.append(32)

        stage1 = []
        stage1.append(InvertedBottleneckBlock(32, 16, 1, stride=1))
        self.stage1 = nn.ModuleList(stage1)
        self._features_ch.append(16)

        stage2 = []
        for i in range(num_layers[1]):
            if i == 0:
                stage2.append(InvertedBottleneckBlock(16, 24, 6, stride=2))
            else:
                stage2.append(InvertedBottleneckBlock(24, 24, 6, stride=1))
        self.stage2 = nn.ModuleList(stage2)
        self._features_ch.append(24)

        stage3 = []
        for i in range(num_layers[2]):
            if i == 0:
                stage3.append(InvertedBottleneckBlock(24, 32, 6, stride=2))
            else:
                stage3.append(InvertedBottleneckBlock(32, 32, 6, stride=1))
        self.stage3 = nn.ModuleList(stage3)
        self._features_ch.append(32)

        stage4 = []
        for i in range(num_layers[3]):
            if i == 0:
                stage4.append(InvertedBottleneckBlock(32, 64, 6, stride=2))
            else:
                stage4.append(InvertedBottleneckBlock(64, 64, 6, stride=1))
        self.stage4 = nn.ModuleList(stage4)
        self._features_ch.append(64)

        stage5 = []
        for i in range(num_layers[4]):
            if i == 0:
                stage5.append(InvertedBottleneckBlock(64, 96, 6, stride=1))
            else:
                stage5.append(InvertedBottleneckBlock(96, 96, 6, stride=1))
        self.stage5 = nn.ModuleList(stage5)
        self._features_ch.append(96)

        stage6 = []
        for i in range(num_layers[5]):
            if i == 0:
                stage6.append(InvertedBottleneckBlock(96, 160, 6, stride=2))
            else:
                stage6.append(InvertedBottleneckBlock(160, 160, 6, stride=1))
        self.stage6 = nn.ModuleList(stage6)
        self._features_ch.append(160)

        stage7 = []
        for i in range(num_layers[6]):
            if i == 0:
                stage7.append(InvertedBottleneckBlock(160, 320, 6, stride=1))
            else:
                stage7.append(InvertedBottleneckBlock(320, 320, 6, stride=1))
        self.stage7 = nn.ModuleList(stage7)
        self._features_ch.append(320)

        stage8 = []
        stage8.append(MobileNetV2OutBlock(320, 1280))
        self.stage8 = nn.ModuleList(stage8)
        self._features_ch.append(1280)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.stem(x)  # (3, 224, 224) -> (32, 112, 112)
        features.append(x)

        for block in self.stage1:  # (32, 112, 112) -> (16, 112, 112)
            x = block(x)
        features.append(x)

        for block in self.stage2:  # (16, 112, 112) -> (24, 56, 56)
            x = block(x)
        features.append(x)

        for block in self.stage3:  # (24, 56, 56) -> (32, 28, 28)
            x = block(x)
        features.append(x)

        for block in self.stage4:  # (32, 28, 28) -> (64, 14, 14)
            x = block(x)
        features.append(x)

        for block in self.stage5:  # (64, 14, 14) -> (96, 14, 14)
            x = block(x)
        features.append(x)

        for block in self.stage6:  # (96, 14, 14) -> (160, 7, 7)
            x = block(x)
        features.append(x)

        for block in self.stage7:  # (160, 7, 7) -> (320, 7, 7)
            x = block(x)
        features.append(x)

        for block in self.stage8:  # (320, 7, 7) -> (1280, 7, 7)
            x = block(x)
        features.append(x)

        return features


@register_vision_backbone("mobilenet_v2")
class MobileNetV2Backbone(_MobileNetV2Base):

    def __init__(self):
        super().__init__(num_layers=[1, 2, 3, 4, 3, 3, 1])
