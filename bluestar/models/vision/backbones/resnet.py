from typing import List
import torch
import torch.nn as nn

from bluestar.modules.convbn import ConvBN2d
from bluestar.modules.squeeze_excite import SqueezeExcite
from .base_backbone import VisionBackboneBase, register_vision_backbone

__all__ = [
    "ResNetStemBlock", "ResNetBasicBlock", "ResNetBottleneckBlock", "ResNet6Backbone", "ResNet9Backbone",
    "ResNet18Backbone", "ResNet34Backbone", "ResNet50Backbone", "ResNet101Backbone", "ResNet152Backbone",
    "ResNeXt50_32x4d_Backbone", "ResNeXt101_32x4d_Backbone", "ResNeXt101_32x8d_Backbone",
]


class ResNetStemBlock(nn.Module):

    def __init__(self, ch: int = 64, manual_batchnorm: bool = False):
        super().__init__()
        self.ch = ch
        self.conv = ConvBN2d(3, ch, kernel_size=7, stride=2, padding=3, manual_batchnorm=manual_batchnorm)
        self.act = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ResNetStemBlock forward.
        :param x:       (batch_size, 3, h, w)
        :return:        (batch_size, ch, h/4, w/4)
        """
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class ResNetBasicBlock(nn.Module):

    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            stride: int = 1,
            manual_batchnorm: bool = False
    ):
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.stride = stride

        self.conv1 = ConvBN2d(input_ch, output_ch, kernel_size=3, stride=stride, padding=1, manual_batchnorm=manual_batchnorm)
        self.act = nn.ReLU(inplace=False)
        self.conv2 = ConvBN2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1, manual_batchnorm=manual_batchnorm)

        if (stride == 1) and (input_ch == output_ch):
            self.down = None
        else:  # cannot directly add identity
            self.down = ConvBN2d(input_ch, output_ch, kernel_size=1, stride=stride, padding=0, manual_batchnorm=manual_batchnorm)

        self.act_out = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ResNetBasicBlock forward
        :param x:   (batch_size, output_ch, h, w)
        :return:    (batch_size, output_ch, h/s, w/s)
        """
        identity = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        if self.down is not None:
            identity = self.down(identity)

        x = x + identity
        x = self.act_out(x)
        return x


class ResNetBottleneckBlock(nn.Module):

    def __init__(
            self,
            input_ch: int,
            middle_ch: int,
            output_ch: int,
            stride: int = 1,
            *, groups: int = 1,
            use_se: bool = False,
            se_ratio: float = 16.0,
            manual_batchnorm: bool = False
    ):
        super().__init__()
        self.input_ch = input_ch
        self.middle_ch = middle_ch
        self.output_ch = output_ch
        self.stride = stride
        self.groups = groups

        self.conv1 = ConvBN2d(input_ch, middle_ch, kernel_size=1, stride=1, padding=0, manual_batchnorm=manual_batchnorm)
        self.act1 = nn.ReLU(inplace=False)
        self.conv2 = ConvBN2d(middle_ch, middle_ch, kernel_size=3, stride=stride, padding=1, groups=groups, manual_batchnorm=manual_batchnorm)
        self.act2 = nn.ReLU(inplace=False)
        self.conv3 = ConvBN2d(middle_ch, output_ch, kernel_size=1, stride=1, padding=0, manual_batchnorm=manual_batchnorm)

        if use_se:
            self.se = SqueezeExcite(output_ch, int(output_ch / se_ratio))
        else:
            self.se = None

        if (stride == 1) and (input_ch == output_ch):
            self.down = None
        else:  # cannot directly add identity
            self.down = ConvBN2d(input_ch, output_ch, kernel_size=1, stride=stride, padding=0, manual_batchnorm=manual_batchnorm)

        self.act_out = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ResNetBottleneckBlock forward
        :param x:   (batch_size, output_ch, h, w)
        :return:    (batch_size, output_ch, h/s, w/s)
        """
        identity = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        if self.se is not None:
            x = self.se(x)

        if self.down is not None:
            identity = self.down(identity)

        x = x + identity
        x = self.act_out(x)
        return x


class ResNetWithBasicBlock(VisionBackboneBase):

    def __init__(self, num_layers: List[int], manual_batchnorm=False):
        super().__init__()
        if len(num_layers) != 4:
            raise ValueError(f"[ERROR] ResNet needs four #layers, got {num_layers}.")
        self.num_layers = num_layers

        self.stem = ResNetStemBlock(ch=64, manual_batchnorm=manual_batchnorm)
        self._features_ch.append(64)

        stage1 = []
        for i in range(num_layers[0]):
            stage1.append(ResNetBasicBlock(64, 64, stride=1, manual_batchnorm=manual_batchnorm))
        self.stage1 = nn.ModuleList(stage1)
        self._features_ch.append(64)

        stage2 = []
        for i in range(num_layers[1]):
            if i == 0:
                stage2.append(ResNetBasicBlock(64, 128, stride=2, manual_batchnorm=manual_batchnorm))
            else:
                stage2.append(ResNetBasicBlock(128, 128, stride=1, manual_batchnorm=manual_batchnorm))
        self.stage2 = nn.ModuleList(stage2)
        self._features_ch.append(128)

        stage3 = []
        for i in range(num_layers[2]):
            if i == 0:
                stage3.append(ResNetBasicBlock(128, 256, stride=2, manual_batchnorm=manual_batchnorm))
            else:
                stage3.append(ResNetBasicBlock(256, 256, stride=1, manual_batchnorm=manual_batchnorm))
        self.stage3 = nn.ModuleList(stage3)
        self._features_ch.append(256)

        stage4 = []
        for i in range(num_layers[3]):
            if i == 0:
                stage4.append(ResNetBasicBlock(256, 512, stride=2, manual_batchnorm=manual_batchnorm))
            else:
                stage4.append(ResNetBasicBlock(512, 512, stride=1, manual_batchnorm=manual_batchnorm))
        self.stage4 = nn.ModuleList(stage4)
        self._features_ch.append(512)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        x = self.stem(x)  # (3, 224, 224) -> (64, 56, 56)

        for block in self.stage1:  # (64, 56, 56) -> (128, 56, 56)
            x = block(x)

        for block in self.stage2:  # (128, 56, 56) -> (256, 28, 28)
            x = block(x)

        for block in self.stage3:  # (256, 28, 28) -> (512, 14, 14)
            x = block(x)

        for block in self.stage4:  # (512, 14, 14) -> (1024, 7, 7)
            x = block(x)

        return x

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.stem(x)  # (3, 224, 224) -> (64, 56, 56)
        features.append(x)

        for block in self.stage1:  # (64, 56, 56) -> (128, 56, 56)
            x = block(x)
        features.append(x)

        for block in self.stage2:  # (128, 56, 56) -> (256, 28, 28)
            x = block(x)
        features.append(x)

        for block in self.stage3:  # (256, 28, 28) -> (512, 14, 14)
            x = block(x)
        features.append(x)

        for block in self.stage4:  # (512, 14, 14) -> (1024, 7, 7)
            x = block(x)
        features.append(x)

        return features


class ResNetWithBottleneckBlock(VisionBackboneBase):

    def __init__(
            self,
            num_layers: List[int],
            groups: int = 1,
            per_group_ch: int = 64,
            use_se: bool = False,
            se_ratio: float = 16.0,
            manual_batchnorm: bool = False
    ) -> None:
        super().__init__()
        if len(num_layers) != 4:
            raise ValueError(f"[ERROR] ResNet needs four #layers, got {num_layers}.")
        self.num_layers = num_layers

        self.stem = ResNetStemBlock(ch=64, manual_batchnorm=manual_batchnorm)
        self._features_ch.append(64)

        ch = groups * per_group_ch
        block_kwargs = dict(groups=groups, use_se=use_se, se_ratio=se_ratio)

        stage1 = []
        for i in range(num_layers[0]):
            if i == 0:
                stage1.append(ResNetBottleneckBlock(64, ch, 256, stride=1, manual_batchnorm=manual_batchnorm, **block_kwargs))
            else:
                stage1.append(ResNetBottleneckBlock(256, ch, 256, stride=1, manual_batchnorm=manual_batchnorm, **block_kwargs))
        self.stage1 = nn.ModuleList(stage1)
        self._features_ch.append(256)

        stage2 = []
        for i in range(num_layers[1]):
            if i == 0:
                stage2.append(ResNetBottleneckBlock(256, ch * 2, 512, stride=2, manual_batchnorm=manual_batchnorm, **block_kwargs))
            else:
                stage2.append(ResNetBottleneckBlock(512, ch * 2, 512, stride=1, manual_batchnorm=manual_batchnorm, **block_kwargs))
        self.stage2 = nn.ModuleList(stage2)
        self._features_ch.append(512)

        stage3 = []
        for i in range(num_layers[2]):
            if i == 0:
                stage3.append(ResNetBottleneckBlock(512, ch * 4, 1024, stride=2, manual_batchnorm=manual_batchnorm, **block_kwargs))
            else:
                stage3.append(ResNetBottleneckBlock(1024, ch * 4, 1024, stride=1, manual_batchnorm=manual_batchnorm, **block_kwargs))
        self.stage3 = nn.ModuleList(stage3)
        self._features_ch.append(1024)

        stage4 = []
        for i in range(num_layers[3]):
            if i == 0:
                stage4.append(ResNetBottleneckBlock(1024, ch * 8, 2048, stride=2, manual_batchnorm=manual_batchnorm, **block_kwargs))
            else:
                stage4.append(ResNetBottleneckBlock(2048, ch * 8, 2048, stride=1, manual_batchnorm=manual_batchnorm, **block_kwargs))
        self.stage4 = nn.ModuleList(stage4)
        self._features_ch.append(2048)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        x = self.stem(x)  # (3, 224, 224) -> (64, 56, 56)

        for block in self.stage1:  # (64, 56, 56) -> (256, 56, 56)
            x = block(x)

        for block in self.stage2:  # (256, 56, 56) -> (512, 28, 28)
            x = block(x)

        for block in self.stage3:  # (512, 28, 28) -> (1024, 14, 14)
            x = block(x)

        for block in self.stage4:  # (1024, 14, 14) -> (2048, 7, 7)
            x = block(x)

        return x

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

        return features

@register_vision_backbone("resnet6")
class ResNet6Backbone(ResNetWithBasicBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[1, 1, 1, 0], manual_batchnorm=manual_batchnorm)

@register_vision_backbone("resnet9")
class ResNet9Backbone(ResNetWithBasicBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[1, 1, 1, 1], manual_batchnorm=manual_batchnorm)


@register_vision_backbone("resnet18")
class ResNet18Backbone(ResNetWithBasicBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[2, 2, 2, 2], manual_batchnorm=manual_batchnorm)


@register_vision_backbone("resnet34")
class ResNet34Backbone(ResNetWithBasicBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[3, 4, 6, 3], manual_batchnorm=manual_batchnorm)


@register_vision_backbone("resnet50")
class ResNet50Backbone(ResNetWithBottleneckBlock):
    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[3, 4, 6, 3], manual_batchnorm=manual_batchnorm)


@register_vision_backbone("resnet101")
class ResNet101Backbone(ResNetWithBottleneckBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[3, 4, 23, 3], manual_batchnorm=manual_batchnorm)


@register_vision_backbone("resnet152")
class ResNet152Backbone(ResNetWithBottleneckBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[3, 8, 36, 3], manual_batchnorm=manual_batchnorm)


@register_vision_backbone("se_resnet50")
class SEResNet50Backbone(ResNetWithBottleneckBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[3, 4, 6, 3], use_se=True, se_ratio=8.0, manual_batchnorm=manual_batchnorm)


@register_vision_backbone("se_resnet101")
class SEResNet101Backbone(ResNetWithBottleneckBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[3, 4, 23, 3], use_se=True, se_ratio=8.0, manual_batchnorm=manual_batchnorm)


@register_vision_backbone("se_resnet152")
class SEResNet152Backbone(ResNetWithBottleneckBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[3, 8, 36, 3], use_se=True, se_ratio=8.0, manual_batchnorm=manual_batchnorm)


@register_vision_backbone("resnext50_32x4d")
class ResNeXt50_32x4d_Backbone(ResNetWithBottleneckBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[3, 4, 6, 3], groups=32, per_group_ch=4, manual_batchnorm=manual_batchnorm)


@register_vision_backbone("resnext101_32x4d")
class ResNeXt101_32x4d_Backbone(ResNetWithBottleneckBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[3, 4, 23, 3], groups=32, per_group_ch=4, manual_batchnorm=manual_batchnorm)


@register_vision_backbone("resnext101_32x8d")
class ResNeXt101_32x8d_Backbone(ResNetWithBottleneckBlock):

    def __init__(self, manual_batchnorm: bool = False):
        super().__init__(num_layers=[3, 4, 23, 3], groups=32, per_group_ch=8, manual_batchnorm=manual_batchnorm)
