from typing import List
import torch
import torch.nn as nn

from bluestar.modules.std_conv import StdConv2d
from .base_backbone import VisionBackboneBase, register_vision_backbone

__all__ = [
    "BiTStemBlock", "BiTBottleneckBlock",
    "BiT50x1Backbone", "BiT50x3Backbone", "BiT101x1Backbone", "BiT101x3Backbone",
    "BiT152x2Backbone", "BiT152x4Backbone",
]


class BiTStemBlock(nn.Module):

    def __init__(self, ch: int = 64):
        super().__init__()
        self.ch = ch
        self.conv = StdConv2d(3, ch, 7, 2, 3, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """BiTStemBlock forward.
        :param x:       (batch_size, 3, h, w)
        :return:        (batch_size, ch, h/4, w/4)
        """
        x = self.conv(x)
        x = self.pool(x)
        return x


class BiTBottleneckBlock(nn.Module):

    def __init__(
            self,
            input_ch: int,
            middle_ch: int,
            output_ch: int,
            stride: int = 1,
    ) -> None:
        super().__init__()
        self.input_ch = input_ch
        self.middle_ch = middle_ch
        self.output_ch = output_ch
        self.stride = stride

        self.gn1 = nn.GroupNorm(32, input_ch)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = StdConv2d(input_ch, middle_ch, 1, 1, 0, bias=False)

        self.gn2 = nn.GroupNorm(32, middle_ch)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = StdConv2d(middle_ch, middle_ch, 3, stride, 1, bias=False)

        self.gn3 = nn.GroupNorm(32, middle_ch)
        self.act3 = nn.ReLU(inplace=True)
        self.conv3 = StdConv2d(middle_ch, output_ch, 1, 1, 0, bias=False)

        if (stride == 1) and (input_ch == output_ch):
            self.down = None
        else:  # cannot directly add identity
            self.down = StdConv2d(input_ch, output_ch, 1, stride, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """BiTBottleneckBlock forward, pre-act bottleneck.
        :param x:   (batch_size, input_ch, h, w)
        :return:    (batch_size, output_ch, h/s, w/s)
        """
        x = self.gn1(x)
        x = self.act1(x)
        identity = x

        x = self.conv1(x)
        x = self.gn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = self.gn3(x)
        x = self.act3(x)
        x = self.conv3(x)

        if self.down is not None:
            identity = self.down(identity)

        x = x + identity
        return x


class BitBackbone(VisionBackboneBase):

    def __init__(self, num_layers: List[int], width_multiplier: int = 1):
        super().__init__()
        if len(num_layers) != 4:
            raise ValueError(f"[ERROR] BiT needs four #layers, got {num_layers}.")
        self.num_layers = num_layers
        self.width_multiplier = width_multiplier

        wm = width_multiplier

        self.stem = BiTStemBlock(ch=64 * wm)
        self._features_ch.append(64 * wm)

        stage1 = []
        for i in range(num_layers[0]):
            if i == 0:
                stage1.append(BiTBottleneckBlock(64 * wm, 64 * wm, 256 * wm, stride=1))
            else:
                stage1.append(BiTBottleneckBlock(256 * wm, 64 * wm, 256 * wm, stride=1))
        self.stage1 = nn.ModuleList(stage1)
        self._features_ch.append(256 * wm)

        stage2 = []
        for i in range(num_layers[1]):
            if i == 0:
                stage2.append(BiTBottleneckBlock(256 * wm, 128 * wm, 512 * wm, stride=2))
            else:
                stage2.append(BiTBottleneckBlock(512 * wm, 128 * wm, 512 * wm, stride=1))
        self.stage2 = nn.ModuleList(stage2)
        self._features_ch.append(512 * wm)

        stage3 = []
        for i in range(num_layers[2]):
            if i == 0:
                stage3.append(BiTBottleneckBlock(512 * wm, 256 * wm, 1024 * wm, stride=2))
            else:
                stage3.append(BiTBottleneckBlock(1024 * wm, 256 * wm, 1024 * wm, stride=1))
        self.stage3 = nn.ModuleList(stage3)
        self._features_ch.append(1024 * wm)

        stage4 = []
        for i in range(num_layers[3]):
            if i == 0:
                stage4.append(BiTBottleneckBlock(1024 * wm, 512 * wm, 2048 * wm, stride=2))
            else:
                stage4.append(BiTBottleneckBlock(2048 * wm, 512 * wm, 2048 * wm, stride=1))
        self.stage4 = nn.ModuleList(stage4)
        self._features_ch.append(2048 * wm)

        self.post_norm = nn.Sequential(
            nn.GroupNorm(32, 2048 * wm),
            nn.ReLU(inplace=False)
        )
        self._features_ch.append(2048 * wm)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.stem(x)  # (3, 224, 224) -> (64, 56, 56)
        features.append(x)

        for block in self.stage1:  # (64, 56, 56) -> (256, 56, 56)
            x = block(x)
        features.append(x)

        for block in self.stage2:  # (256, 56, 56) -> (512, 28, 28)
            x = block(x)
        features.append(x)

        for block in self.stage3:  # (512, 28, 28) -> (1024, 14, 14)
            x = block(x)
        features.append(x)

        for block in self.stage4:  # (1024, 14, 14) -> (2048, 7, 7)
            x = block(x)
        features.append(x)

        x = self.post_norm(x)
        features.append(x)

        return features


@register_vision_backbone("bit50x1")
class BiT50x1Backbone(BitBackbone):

    def __init__(self):
        super().__init__(num_layers=[3, 4, 6, 3], width_multiplier=1)


@register_vision_backbone("bit50x3")
class BiT50x3Backbone(BitBackbone):

    def __init__(self):
        super().__init__(num_layers=[3, 4, 6, 3], width_multiplier=3)


@register_vision_backbone("bit101x1")
class BiT101x1Backbone(BitBackbone):

    def __init__(self):
        super().__init__(num_layers=[3, 4, 23, 3], width_multiplier=1)


@register_vision_backbone("bit101x3")
class BiT101x3Backbone(BitBackbone):

    def __init__(self):
        super().__init__(num_layers=[3, 4, 23, 3], width_multiplier=3)


@register_vision_backbone("bit152x2")
class BiT152x2Backbone(BitBackbone):

    def __init__(self):
        super().__init__(num_layers=[3, 8, 36, 3], width_multiplier=2)


@register_vision_backbone("bit152x4")
class BiT152x4Backbone(BitBackbone):

    def __init__(self):
        super().__init__(num_layers=[3, 8, 36, 3], width_multiplier=4)
