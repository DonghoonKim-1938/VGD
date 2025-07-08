"""
Implementation for "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML, 2019)
https://proceedings.mlr.press/v97/tan19a/tan19a.pdf
by DH Kim
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np

from bluestar.modules.convbn import ConvBN2d
from bluestar.modules.depthwise_conv import DepthwiseConvBN2d
from bluestar.modules.pointwise_conv import PointwiseConvBN2d
from bluestar.modules.squeeze_excite import SqueezeExcite
from bluestar.modules.drop_path import DropPath
from bluestar.models.utils import _make_divisible
from .base_backbone import VisionBackboneBase, register_vision_backbone

__all__ = [
    # EfficientNet blocks
    "MBConvBlock",
    "EfficientNetBase",
    # EfficientNet models
    "EfficientNetB0Backbone",
    "EfficientNetB1Backbone",
    "EfficientNetB2Backbone",
    "EfficientNetB3Backbone",
    "EfficientNetB4Backbone",
    "EfficientNetB5Backbone",
    "EfficientNetB6Backbone",
    "EfficientNetB7Backbone",
    "EfficientNetB8Backbone",
    "EfficientNetL2Backbone",
]


class EfficientNetStemBlock(nn.Module):

    def __init__(self, ch: int = 32):
        super().__init__()
        self.conv = ConvBN2d(3, ch, kernel_size=3, stride=2, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EfficientNetStemBlock forward.
        :param x:       (batch_size, 3, h, w)
        :return:        (batch_size, ch, h/2, w/2)
        """
        x = self.conv(x)
        x = self.act(x)
        return x


class MBConvBlock(nn.Module):
    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            expand_ratio: int = 1,
            kernel_size: int = 3,
            stride: int = 1,
            se_ratio: int = 4,
            drop_prob: float = 0.0,
    ):
        super().__init__()
        self.stride = stride
        if self.stride != 1 and self.stride != 2:
            raise ValueError(f"[ERROR] MBConvBlock needs stride 1 or 2 . {stride} is not supported.")
        self.has_skip_connection = (self.stride == 1) and (input_ch == output_ch)

        # Pointwise convolution: does not apply for the first layer
        if expand_ratio > 1:
            self.conv0 = PointwiseConvBN2d(input_ch, input_ch * expand_ratio)
            self.act0 = nn.SiLU(inplace=True)
        else:
            self.conv0 = self.act0 = None
        middle_ch = input_ch * expand_ratio

        # Depthwise convolution
        self.conv1 = DepthwiseConvBN2d(middle_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.act1 = nn.SiLU(inplace=True)

        # Squeeze and Execution (SE)
        self.se = SqueezeExcite(middle_ch, input_ch // se_ratio, activation_layer=nn.SiLU)

        # Pointwise convolution
        self.conv2 = PointwiseConvBN2d(middle_ch, output_ch)
        self.act2 = nn.SiLU(inplace=True)

        # Drop Connection
        self.drop_path = DropPath(drop_prob=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MBConvBlock forward
        :param x:   (batch_size, input_ch, h, w)
        :return:    (batch_size, output_ch, h/s, w/s)
        """
        identity = x
        if self.conv0 is not None:
            x = self.conv0(x)
            x = self.act0(x)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.se(x)
        x = self.conv2(x)
        x = self.act2(x)

        # Residual
        if self.has_skip_connection:
            x = self.drop_path(x)  # whether we just pass this layer or not
            x = x + identity
        return x


class EfficientNetBase(VisionBackboneBase):
    def __init__(
            self,
            alpha: float,  # alpha:  tuning the depth of the architecture
            beta: float,  # beta: tuning the width of the architecture
            gamma: float,  # gamma: input image resolution
            drop_prob: float,  # dropout probability
    ) -> None:
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma  # keep

        # base layers and channels
        self.num_layers, self.channels = self.scale_network(alpha, beta)

        # (B0): (3, 224, 224) -> (32, 112, 112)
        self.stem = EfficientNetStemBlock(ch=self.channels[0])
        self._features_ch.append(self.channels[0])

        total_blocks = sum(self.num_layers[1:-1])
        block_id = 0

        # (B0): (32, 112, 112) -> (16, 112, 112)
        stage1 = []
        for i in range(self.num_layers[1]):
            if i == 0:
                stage1.append(
                    MBConvBlock(
                        input_ch=self.channels[0],
                        output_ch=self.channels[1],
                        kernel_size=3,
                        stride=1,
                        se_ratio=4,
                        expand_ratio=1,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            else:
                stage1.append(
                    MBConvBlock(
                        input_ch=self.channels[1],
                        output_ch=self.channels[1],
                        kernel_size=3,
                        stride=1,
                        se_ratio=4,
                        expand_ratio=1,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            block_id += 1
        self.stage1 = nn.ModuleList(stage1)
        self._features_ch.append(self.channels[1])

        # (B0): (16, 112, 112) -> (24, 56, 56)
        stage2 = []
        for i in range(self.num_layers[2]):
            if i == 0:
                stage2.append(
                    MBConvBlock(
                        input_ch=self.channels[1],
                        output_ch=self.channels[2],
                        kernel_size=3,
                        stride=2,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            else:
                stage2.append(
                    MBConvBlock(
                        input_ch=self.channels[2],
                        output_ch=self.channels[2],
                        kernel_size=3,
                        stride=1,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            block_id += 1
        self.stage2 = nn.ModuleList(stage2)
        self._features_ch.append(self.channels[2])

        # (B0): (24, 56, 56) -> (40, 28, 28)
        stage3 = []
        for i in range(self.num_layers[3]):
            if i == 0:
                stage3.append(
                    MBConvBlock(
                        input_ch=self.channels[2],
                        output_ch=self.channels[3],
                        kernel_size=5,
                        stride=2,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            else:
                stage3.append(
                    MBConvBlock(
                        input_ch=self.channels[3],
                        output_ch=self.channels[3],
                        kernel_size=5,
                        stride=1,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            block_id += 1
        self.stage3 = nn.ModuleList(stage3)
        self._features_ch.append(self.channels[3])

        # (B0): (40, 28, 28) -> (80, 28, 28)
        stage4 = []
        for i in range(self.num_layers[4]):
            if i == 0:
                stage4.append(
                    MBConvBlock(
                        input_ch=self.channels[3],
                        output_ch=self.channels[4],
                        kernel_size=3,
                        stride=1,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            else:
                stage4.append(
                    MBConvBlock(
                        input_ch=self.channels[4],
                        output_ch=self.channels[4],
                        kernel_size=3,
                        stride=1,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            block_id += 1
        self.stage4 = nn.ModuleList(stage4)
        self._features_ch.append(self.channels[4])

        # (B0): (80, 28, 28) -> (112, 14, 14)
        stage5 = []
        for i in range(self.num_layers[5]):
            if i == 0:
                stage5.append(
                    MBConvBlock(
                        input_ch=self.channels[4],
                        output_ch=self.channels[5],
                        kernel_size=5,
                        stride=2,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            else:
                stage5.append(
                    MBConvBlock(
                        input_ch=self.channels[5],
                        output_ch=self.channels[5],
                        kernel_size=5,
                        stride=1,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            block_id += 1
        self.stage5 = nn.ModuleList(stage5)
        self._features_ch.append(self.channels[5])

        # (B0): (112, 14, 14) -> (192, 7, 7)
        stage6 = []
        for i in range(self.num_layers[6]):
            if i == 0:
                stage6.append(
                    MBConvBlock(
                        input_ch=self.channels[5],
                        output_ch=self.channels[6],
                        kernel_size=5,
                        stride=2,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            else:
                stage6.append(
                    MBConvBlock(
                        input_ch=self.channels[6],
                        output_ch=self.channels[6],
                        kernel_size=5,
                        stride=1,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            block_id += 1
        self.stage6 = nn.ModuleList(stage6)
        self._features_ch.append(self.channels[6])

        # (B0): (192, 7, 7) -> (320, 7, 7)
        stage7 = []
        for i in range(self.num_layers[7]):
            if i == 0:
                stage7.append(
                    MBConvBlock(
                        input_ch=self.channels[6],
                        output_ch=self.channels[7],
                        kernel_size=3,
                        stride=1,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            else:
                stage7.append(
                    MBConvBlock(
                        input_ch=self.channels[7],
                        output_ch=self.channels[7],
                        kernel_size=3,
                        stride=1,
                        se_ratio=4,
                        expand_ratio=6,
                        drop_prob=drop_prob * float(block_id) / total_blocks
                    )
                )
            block_id += 1
        self.stage7 = nn.ModuleList(stage7)
        self._features_ch.append(self.channels[7])

        # (B0): (320, 7, 7) -> (1280, 7, 7)
        stage8 = []
        stage8.append(ConvBN2d(
            in_channels=self.channels[7],
            out_channels=self.channels[8],
            kernel_size=1,
            stride=1,
        ))
        stage8.append(nn.SiLU(inplace=True))
        self.stage8 = nn.ModuleList(stage8)
        self._features_ch.append(self.channels[8])

    @staticmethod
    def scale_network(alpha: float, beta: float) -> Tuple[List[int], List[int]]:
        # alpha: tuning the depth of the architecture
        # beta: tuning the width of the architecture
        num_layers0 = np.array([1, 1, 2, 2, 3, 3, 4, 1, 1], dtype=np.float32)  # B0
        channels0 = np.array([32, 16, 24, 40, 80, 112, 192, 320, 1280], dtype=np.float32)  # B0

        # scaling the depth (layers) of the network
        num_layers = num_layers0.copy()
        num_layers[1:-1] *= alpha
        num_layers = list(np.ceil(num_layers).astype(int).tolist())

        # scaling the width (channels) of the network
        channels = channels0.copy()
        channels[:-1] *= beta
        channels = list(channels.tolist())
        # make channels multiple of 8
        channels = [_make_divisible(ch, divisor=8) for ch in channels]
        channels[-1] = max(1280, channels[-2] * 4)
        return num_layers, channels

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.stem(x)
        features.append(x)

        for block in self.stage1:
            x = block(x)
        features.append(x)

        for block in self.stage2:
            x = block(x)
        features.append(x)

        for block in self.stage3:
            x = block(x)
        features.append(x)

        for block in self.stage4:
            x = block(x)
        features.append(x)

        for block in self.stage5:
            x = block(x)
        features.append(x)

        for block in self.stage6:
            x = block(x)
        features.append(x)

        for block in self.stage7:
            x = block(x)
        features.append(x)

        for block in self.stage8:
            x = block(x)
        features.append(x)

        return features


_model_scale_dict = {
    # (depth_coefficient, width_coefficient, resolution, dropout_rate)
    'efficientnet_b0': {'alpha': 1.0, 'beta': 1.0, 'gamma': 224, 'drop_prob': 0.2},
    'efficientnet_b1': {'alpha': 1.1, 'beta': 1.0, 'gamma': 240, 'drop_prob': 0.2},
    'efficientnet_b2': {'alpha': 1.2, 'beta': 1.1, 'gamma': 260, 'drop_prob': 0.3},
    'efficientnet_b3': {'alpha': 1.4, 'beta': 1.2, 'gamma': 300, 'drop_prob': 0.3},
    'efficientnet_b4': {'alpha': 1.8, 'beta': 1.4, 'gamma': 380, 'drop_prob': 0.4},
    'efficientnet_b5': {'alpha': 2.2, 'beta': 1.6, 'gamma': 456, 'drop_prob': 0.4},
    'efficientnet_b6': {'alpha': 2.6, 'beta': 1.8, 'gamma': 528, 'drop_prob': 0.5},
    'efficientnet_b7': {'alpha': 3.1, 'beta': 2.0, 'gamma': 600, 'drop_prob': 0.5},
    'efficientnet_b8': {'alpha': 3.6, 'beta': 2.2, 'gamma': 672, 'drop_prob': 0.5},
    'efficientnet_l2': {'alpha': 5.3, 'beta': 4.3, 'gamma': 800, 'drop_prob': 0.5},
}


@register_vision_backbone("efficientnet_b0")
class EfficientNetB0Backbone(EfficientNetBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['efficientnet_b0'])


@register_vision_backbone("efficientnet_b1")
class EfficientNetB1Backbone(EfficientNetBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['efficientnet_b1'])


@register_vision_backbone("efficientnet_b2")
class EfficientNetB2Backbone(EfficientNetBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['efficientnet_b2'])


@register_vision_backbone("efficientnet_b3")
class EfficientNetB3Backbone(EfficientNetBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['efficientnet_b3'])


@register_vision_backbone("efficientnet_b4")
class EfficientNetB4Backbone(EfficientNetBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['efficientnet_b4'])


@register_vision_backbone("efficientnet_b5")
class EfficientNetB5Backbone(EfficientNetBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['efficientnet_b5'])


@register_vision_backbone("efficientnet_b6")
class EfficientNetB6Backbone(EfficientNetBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['efficientnet_b6'])


@register_vision_backbone("efficientnet_b7")
class EfficientNetB7Backbone(EfficientNetBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['efficientnet_b7'])


@register_vision_backbone("efficientnet_b8")
class EfficientNetB8Backbone(EfficientNetBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['efficientnet_b8'])


@register_vision_backbone("efficientnet_l2")
class EfficientNetL2Backbone(EfficientNetBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['efficientnet_l2'])
