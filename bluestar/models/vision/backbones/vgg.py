from typing import List
import torch
import torch.nn as nn

from bluestar.modules.convbn import ConvBN2d
from .base_backbone import VisionBackboneBase, register_vision_backbone

__all__ = [
    "VGGBlock", "VGG11Backbone", "VGG13Backbone", "VGG16Backbone", "VGG19Backbone"
]


class VGGBlock(nn.Module):

    def __init__(
            self,
            input_ch: int,
            output_ch: int
    ) -> None:
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.conv = ConvBN2d(self.input_ch, self.output_ch, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """VGGBBlock forward
        :param x:   (batch_size, input_ch, h, w)
        :return:    (batch_size, output_ch, h, w)
        """
        x = self.conv(x)
        x = self.act(x)
        return x


class VGGBackbone(VisionBackboneBase):

    def __init__(self, num_layers: List[int]):
        super().__init__()
        if len(num_layers) != 5:
            raise ValueError(f"[ERROR] VGG needs five #layers, got {num_layers}.")
        self.num_layers = num_layers

        stage1 = []
        stage1.append(VGGBlock(3, 64))
        for i in range(num_layers[0] - 1):
            stage1.append(VGGBlock(64, 64))
        stage1.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage1 = nn.ModuleList(stage1)
        self._features_ch.append(64)

        stage2 = []
        stage2.append(VGGBlock(64, 128))
        for i in range(num_layers[1] - 1):
            stage2.append(VGGBlock(128, 128))
        stage2.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage2 = nn.ModuleList(stage2)
        self._features_ch.append(128)

        stage3 = []
        stage3.append(VGGBlock(128, 256))
        for i in range(num_layers[2] - 1):
            stage3.append(VGGBlock(256, 256))
        stage3.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage3 = nn.ModuleList(stage3)
        self._features_ch.append(256)

        stage4 = []
        stage4.append(VGGBlock(256, 512))
        for i in range(num_layers[3] - 1):
            stage4.append(VGGBlock(512, 512))
        stage4.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage4 = nn.ModuleList(stage4)
        self._features_ch.append(512)

        stage5 = []
        for i in range(num_layers[4]):
            stage5.append(VGGBlock(512, 512))
        stage5.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.stage5 = nn.ModuleList(stage5)
        self._features_ch.append(512)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        for block in self.stage1:  # (3, 224, 224) -> (64, 112, 112)
            x = block(x)
        features.append(x)

        for block in self.stage2:  # (64, 112, 112) -> (128, 56, 56)
            x = block(x)
        features.append(x)

        for block in self.stage3:  # (128, 56, 56) -> (256, 28, 28)
            x = block(x)
        features.append(x)

        for block in self.stage4:  # (256, 28, 28) -> (512, 14, 14)
            x = block(x)
        features.append(x)

        for block in self.stage5:  # (512, 14, 14) -> (512, 7, 7)
            x = block(x)
        features.append(x)

        return features


@register_vision_backbone("vgg11")
class VGG11Backbone(VGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[1, 1, 2, 2, 2])


@register_vision_backbone("vgg13")
class VGG13Backbone(VGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[2, 2, 2, 2, 2])


@register_vision_backbone("vgg16")
class VGG16Backbone(VGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[2, 2, 3, 3, 3])


@register_vision_backbone("vgg19")
class VGG19Backbone(VGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[2, 2, 4, 4, 4])
