from typing import List
import torch
import torch.nn as nn

from bluestar.modules.layernorm import LayerNorm2d
from bluestar.modules.drop_path import DropPath
from .base_backbone import VisionBackboneBase, register_vision_backbone

__all__ = [
    "ConvNeXtBlock", "ConvNeXtBase",
    "ConvNeXtTinyBackbone", "ConvNeXtSmallBackbone", "ConvNeXtBaseBackbone", "ConvNeXtLargeBackbone",
]

"""
ConvNeXt improvements
1) Training Techniques : 90 ep -> 300 ep, AdamW, data aug (Mixup, Cutmix, RandAug, Rand Erasing), label smoothing
2) #layers for each stage : (3, 4, 6, 3) -> (3, 3, 9, 3)
3) Patchify (non-overlapping) stem : 4 * 4 filter size + 4 stride
4) Separate into channel groups + depthwise conv + width (64 -> 96)
5) Inverted bottleneck
6) 7 * 7 kernel size for depthwise conv
7) ReLU -> GELU
8) Fewer activation functions : pointwise + GELU + pointwise
9) Fewer normalization layers : LayerNorm + pointwise
10) BN -> LN
11) Patch merging down-sampling: 2 *2 conv stride 2 + normalization layer
"""


class ConvNeXtBlock(nn.Module):
    def __init__(
            self,
            ch: int,
            drop_prob: float,
            init_layer_scale: float = 1e-6,
    ):
        super().__init__()

        self.conv0 = nn.Conv2d(ch, ch, kernel_size=7, stride=1, padding=3, groups=ch, bias=True)
        self.norm = nn.LayerNorm(ch)
        self.conv1 = nn.Linear(ch, ch * 4, bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Linear(ch * 4, ch, bias=True)

        self.path_scale = nn.Parameter(torch.ones(ch, 1, 1) * init_layer_scale)  # initialize with very small value
        self.drop_path = DropPath(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ConvNeXtBlock forward
        :param x:       (batch_size, ch, h, w)
        :return:        (batch_size, ch, h, w)
        """
        identity = x

        x = self.conv0(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, c, h, w) -> (b, h, w, c)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (b, h, w, c) -> (b, c, h, w)

        x = x * self.path_scale
        x = self.drop_path(x)

        x = x + identity
        return x


class ConvNeXtBase(VisionBackboneBase):
    def __init__(
            self,
            num_layers: List[int],  # [3, 3, 9, 3]
            channels: List[int],  # [96, 192, 384, 768]
            drop_prob: float = 0,  # path drop rate
            *, init_layer_scale: float = 1e-6,
            eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(num_layers) != 4:
            raise ValueError(f"[ERROR] ConvNeXt needs four #layers, got {num_layers}.")
        if len(channels) != 4:
            raise ValueError(f"[ERROR] ResNet needs four #channels, got {channels}.")

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=4, stride=4, padding=0, bias=True),
            LayerNorm2d(channels[0], eps=eps)
        )
        self._features_ch.append(channels[0])

        total_blocks = sum(num_layers)
        block_id = 0

        stage1 = []  # 1/4 scale
        for i in range(num_layers[0]):
            stage1.append(ConvNeXtBlock(
                ch=channels[0],
                drop_prob=drop_prob * float(block_id) / total_blocks,
                init_layer_scale=init_layer_scale,
            ))
            block_id += 1
        self.stage1 = nn.ModuleList(stage1)
        self._features_ch.append(channels[0])

        self.downsample1 = nn.Sequential(
            LayerNorm2d(channels[0], eps=eps),
            nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2, padding=0, bias=True)
        )  # 1/4 -> 1/8

        stage2 = []  # 1/8 scale
        for i in range(num_layers[1]):
            stage2.append(ConvNeXtBlock(
                ch=channels[1],
                drop_prob=drop_prob * float(block_id) / total_blocks,
                init_layer_scale=init_layer_scale,
            ))
            block_id += 1
        self.stage2 = nn.ModuleList(stage2)
        self._features_ch.append(channels[1])

        self.downsample2 = nn.Sequential(
            LayerNorm2d(channels[1], eps=eps),
            nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2, padding=0, bias=True)
        )  # 1/8 -> 1/16

        stage3 = []  # 1/16 scale
        for i in range(num_layers[2]):
            stage3.append(ConvNeXtBlock(
                ch=channels[2],
                drop_prob=drop_prob * float(block_id) / total_blocks,
                init_layer_scale=init_layer_scale,
            ))
            block_id += 1
        self.stage3 = nn.ModuleList(stage3)
        self._features_ch.append(channels[2])

        self.downsample3 = nn.Sequential(
            LayerNorm2d(channels[2], eps=eps),
            nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2, padding=0, bias=True)
        )  # 1/16 -> 1/32

        stage4 = []  # 1/32 scale
        for i in range(num_layers[3]):
            stage4.append(ConvNeXtBlock(
                ch=channels[3],
                drop_prob=drop_prob * float(block_id) / total_blocks,
                init_layer_scale=init_layer_scale,
            ))
            block_id += 1
        self.stage4 = nn.ModuleList(stage4)
        self._features_ch.append(channels[3])

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.stem(x)
        features.append(x)

        for block in self.stage1:
            x = block(x)
        features.append(x)

        x = self.downsample1(x)
        for block in self.stage2:
            x = block(x)
        features.append(x)

        x = self.downsample2(x)
        for block in self.stage3:
            x = block(x)
        features.append(x)

        x = self.downsample3(x)
        for block in self.stage4:
            x = block(x)
        features.append(x)

        return features


@register_vision_backbone("convnext_tiny")
class ConvNeXtTinyBackbone(ConvNeXtBase):

    def __init__(self):
        super().__init__(num_layers=[3, 3, 9, 3], channels=[96, 192, 384, 768], drop_prob=0.1)


@register_vision_backbone("convnext_small")
class ConvNeXtSmallBackbone(ConvNeXtBase):

    def __init__(self):
        super().__init__(num_layers=[3, 3, 27, 3], channels=[96, 192, 384, 768], drop_prob=0.4)


@register_vision_backbone("convnext_base")
class ConvNeXtBaseBackbone(ConvNeXtBase):

    def __init__(self):
        super().__init__(num_layers=[3, 3, 27, 3], channels=[128, 256, 512, 1024], drop_prob=0.5)


@register_vision_backbone("convnext_large")
class ConvNeXtLargeBackbone(ConvNeXtBase):

    def __init__(self):
        super().__init__(num_layers=[3, 3, 27, 3], channels=[192, 384, 768, 1536], drop_prob=0.5)
