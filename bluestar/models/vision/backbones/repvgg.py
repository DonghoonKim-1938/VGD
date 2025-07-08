from typing import List
import torch
import torch.nn as nn

from bluestar.modules.repvgg_conv import RepVGGConvBN2d
from bluestar.modules.squeeze_excite import SqueezeExcite
from .base_backbone import VisionBackboneBase, register_vision_backbone

__all__ = [
    "RepVGGBlock", "RepVGGBackbone",
    "RepVGGA0Backbone", "RepVGGA1Backbone", "RepVGGA2Backbone",
    "RepVGGB0Backbone", "RepVGGB1Backbone", "RepVGGB2Backbone", "RepVGGB3Backbone",
    "RepVGGD2seBackbone",
]


class RepVGGBlock(nn.Module):

    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            stride: int = 1,
            *, use_se: bool = False,
            se_ratio: float = 16.0,
    ) -> None:
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.stride = stride

        self.conv = RepVGGConvBN2d(input_ch, output_ch, kernel_size=3, stride=stride, padding=1)
        if use_se:
            self.se = SqueezeExcite(output_ch, int(output_ch / se_ratio))
        else:
            self.se = None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RepVGGBlock forward
        :param x:   (batch_size, input_ch, h, w)
        :return:    (batch_size, output_ch, h/s, w/s)
        """
        x = self.conv(x)
        if self.se is not None:
            x = self.se(x)
        x = self.act(x)
        return x


class RepVGGBackbone(VisionBackboneBase):

    def __init__(
            self, num_layers: List[int],
            width_multiplier_a: float = 1.0,
            width_multiplier_b: float = 1.0,  # only for the last stage
            *, use_se: bool = False,
            se_ratio: float = 16.0,
    ):
        super().__init__()
        if len(num_layers) != 4:
            raise ValueError(f"[ERROR] RepVGG needs four #layers, got {num_layers}.")
        self.num_layers = num_layers
        wma = width_multiplier_a
        wmb = width_multiplier_b

        se_kwargs = dict(use_se=use_se, se_ratio=se_ratio)

        ch = min(64, int(64 * wma))
        self.stem = RepVGGBlock(3, ch, stride=2, **se_kwargs)
        self._features_ch.append(ch)

        stage1 = []
        for i in range(num_layers[0]):
            if i == 0:
                stage1.append(RepVGGBlock(ch, int(64 * wma), stride=2, **se_kwargs))
                ch = int(64 * wma)
            else:
                stage1.append(RepVGGBlock(ch, ch, stride=1, **se_kwargs))
        self.stage1 = nn.ModuleList(stage1)
        self._features_ch.append(ch)

        stage2 = []
        for i in range(num_layers[1]):
            if i == 0:
                stage2.append(RepVGGBlock(ch, int(128 * wma), stride=2, **se_kwargs))
                ch = int(128 * wma)
            else:
                stage2.append(RepVGGBlock(ch, ch, stride=1, **se_kwargs))
        self.stage2 = nn.ModuleList(stage2)
        self._features_ch.append(ch)

        stage3 = []
        for i in range(num_layers[2]):
            if i == 0:
                stage3.append(RepVGGBlock(ch, int(256 * wma), stride=2, **se_kwargs))
                ch = int(256 * wma)
            else:
                stage3.append(RepVGGBlock(ch, ch, stride=1, **se_kwargs))
        self.stage3 = nn.ModuleList(stage3)
        self._features_ch.append(ch)

        stage4 = []
        for i in range(num_layers[3]):
            if i == 0:
                stage4.append(RepVGGBlock(ch, int(512 * wmb), stride=2, **se_kwargs))
                ch = int(512 * wmb)
            else:
                stage4.append(RepVGGBlock(ch, ch, stride=1, **se_kwargs))
        self.stage4 = nn.ModuleList(stage4)
        self._features_ch.append(ch)

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


@register_vision_backbone("repvgg_a0")
class RepVGGA0Backbone(RepVGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[2, 4, 14, 1], width_multiplier_a=0.75, width_multiplier_b=2.5)


@register_vision_backbone("repvgg_a1")
class RepVGGA1Backbone(RepVGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[2, 4, 14, 1], width_multiplier_a=1.0, width_multiplier_b=2.5)


@register_vision_backbone("repvgg_a2")
class RepVGGA2Backbone(RepVGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[2, 4, 14, 1], width_multiplier_a=1.5, width_multiplier_b=2.75)


@register_vision_backbone("repvgg_b0")
class RepVGGB0Backbone(RepVGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[4, 6, 16, 1], width_multiplier_a=1.0, width_multiplier_b=2.5)


@register_vision_backbone("repvgg_b1")
class RepVGGB1Backbone(RepVGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[4, 6, 16, 1], width_multiplier_a=2.0, width_multiplier_b=4.0)


@register_vision_backbone("repvgg_b2")
class RepVGGB2Backbone(RepVGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[4, 6, 16, 1], width_multiplier_a=2.5, width_multiplier_b=5.0)


@register_vision_backbone("repvgg_b3")
class RepVGGB3Backbone(RepVGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[4, 6, 16, 1], width_multiplier_a=3.0, width_multiplier_b=5.0)


@register_vision_backbone("repvgg_d2se")
class RepVGGD2seBackbone(RepVGGBackbone):

    def __init__(self):
        super().__init__(num_layers=[8, 14, 24, 1], width_multiplier_a=2.5, width_multiplier_b=5.0,
                         use_se=True, se_ratio=8.0)
