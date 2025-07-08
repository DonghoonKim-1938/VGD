from typing import List, Optional
import math
import torch
import torch.nn as nn

from bluestar.modules.convbn import ConvBN2d
from bluestar.modules.squeeze_excite import SqueezeExcite
from bluestar.models.utils import _make_divisible
from .base_backbone import VisionBackboneBase, register_vision_backbone

__all__ = [
    "RegNetBottleneckBlock", "RegNet",
    "RegNetX8GFBackbone", "RegNetX16GFBackbone", "RegNetX32GFBackbone",
    "RegNetY8GFBackbone", "RegNetY16GFBackbone", "RegNetY32GFBackbone", "RegNetY128GFBackbone",
]


class RegNetBottleneckBlock(nn.Module):

    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            stride: int,
            per_group_ch: int,
            bottleneck_multiplier: float,
            se_ratio: Optional[float] = None,
            *, activation_layer=nn.ReLU
    ) -> None:
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.stride = stride

        bottleneck_ch = int(round(output_ch * bottleneck_multiplier))
        if bottleneck_ch % per_group_ch != 0:
            raise ValueError(f"RegNetBottleneck channels {bottleneck_ch} not divisible by"
                             f" per-group channels {per_group_ch}.")
        groups = bottleneck_ch // per_group_ch
        self.groups = groups

        layers = [
            ConvBN2d(input_ch, bottleneck_ch, kernel_size=1, stride=1),
            activation_layer(),
            ConvBN2d(bottleneck_ch, bottleneck_ch, kernel_size=3, stride=stride, groups=groups, padding=1),
            activation_layer(),
        ]
        if (se_ratio is not None) and (se_ratio > 0):
            reduce_ch = int(round(se_ratio * input_ch))
            layers.append(SqueezeExcite(bottleneck_ch, reduce_ch, activation_layer=activation_layer))
        layers.append(ConvBN2d(bottleneck_ch, output_ch, kernel_size=1, stride=1))

        self.layers = nn.Sequential(*layers)

        if (stride == 1) and (input_ch == output_ch):
            self.down = None
        else:  # cannot directly add identity
            self.down = ConvBN2d(input_ch, output_ch, kernel_size=1, stride=stride, padding=0)

        self.act_out = activation_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RegNetBottleneckBlock forward
        :param x:   (batch_size, input_ch, h, w)
        :return:    (batch_size, output_ch, h/s, w/s)
        """
        identity = x
        x = self.layers(x)

        if self.down is not None:
            identity = self.down(identity)

        x = x + identity
        x = self.act_out(x)
        return x


class RegNetStage(nn.Sequential):

    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            num_layers: int,
            stride: int,
            per_group_ch: int,
            bottleneck_multiplier: float,
            se_ratio: Optional[float] = None,
            *, activation_layer=nn.ReLU,
            stage_index: int = 0,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            block_name = f"block{stage_index}-{i}"
            block = RegNetBottleneckBlock(
                input_ch=(input_ch if (i == 0) else output_ch),
                output_ch=output_ch,
                stride=(stride if (i == 0) else 1),
                per_group_ch=per_group_ch,
                bottleneck_multiplier=bottleneck_multiplier,
                se_ratio=se_ratio,
                activation_layer=activation_layer
            )
            self.add_module(block_name, block)


class RegNetParams(object):

    def __init__(
            self,
            num_layers: List[int],
            channels: List[int],
            strides: List[int],
            per_group_channels: List[int],
            bottleneck_multipliers: List[float],
            se_ratios: List[Optional[float]],
    ) -> None:
        # recommend not to create these parameters by __init__.
        self.num_stages = len(num_layers)
        self.num_layers = num_layers
        self.channels = channels
        self.strides = strides
        self.per_group_channels = per_group_channels
        self.bottleneck_multipliers = bottleneck_multipliers
        self.se_ratios = se_ratios

    def __len__(self) -> int:
        return self.num_stages

    def param_iter(self):
        return zip(self.num_layers, self.channels, self.strides, self.per_group_channels,
                   self.bottleneck_multipliers, self.se_ratios)

    @classmethod
    def from_init_params(
            cls,
            num_layer: int,
            base_ch: int,
            ch_progress_slope: float,
            ch_log_step: float,
            per_group_ch: int,
            bottleneck_multiplier: float = 1.0,
            se_ratio: float = None,
    ) -> "RegNetParams":
        # ---- determine stage ---- #
        ch_cont = torch.arange(num_layer) * ch_progress_slope + base_ch
        block_capacity = torch.round(torch.log(ch_cont / base_ch) / math.log(ch_log_step))
        block_channels = torch.round(torch.divide(base_ch * torch.pow(ch_log_step, block_capacity), 8) * 8)
        block_channels = block_channels.int().tolist()
        num_stages = len(set(block_channels))

        # ---- per-stage params ---- #
        split_helper = zip(block_channels + [0], [0] + block_channels)
        splits = [(x != xn) for x, xn in split_helper]

        channels = [ch for ch, t in zip(block_channels, splits[:-1]) if t]
        num_layers = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [2] * num_stages
        se_ratios = [se_ratio] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        per_group_channels = [per_group_ch] * num_stages

        # ---- divisible by groups ---- #
        bottleneck_channels = [int(ch * b) for ch, b in zip(channels, bottleneck_multipliers)]
        per_group_channels = [min(per_g, ch) for per_g, ch in zip(per_group_channels, bottleneck_channels)]
        bottleneck_channels = [_make_divisible(ch, per_g) for ch, per_g in zip(bottleneck_channels, per_group_channels)]
        channels = [int(ch / b) for ch, b in zip(bottleneck_channels, bottleneck_multipliers)]

        return cls(num_layers, channels, strides, per_group_channels, bottleneck_multipliers, se_ratios)


class RegNet(VisionBackboneBase):

    def __init__(
            self,
            params: RegNetParams,
            *, activation_layer=nn.ReLU
    ):
        super().__init__()
        self.num_layers = []

        self.stem = nn.Sequential(
            ConvBN2d(3, 32, 3, stride=2, padding=1),
            activation_layer()
        )
        self._features_ch.append(32)

        current_ch = 32
        stages = []
        for i, (num_layer, ch, stride, per_group_ch, bottleneck_mult, se_ratio) in enumerate(params.param_iter()):
            stages.append(RegNetStage(
                current_ch, ch, num_layer, stride, per_group_ch, bottleneck_mult, se_ratio,
                activation_layer=activation_layer, stage_index=i + 1  # after stem
            ))
            self._features_ch.append(ch)
            self.num_layers.append(num_layer)
            current_ch = ch

        self.stages = nn.ModuleList(stages)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        x = self.stem(x)  # (3, 224, 224) -> (32, 112, 112)
        features.append(x)

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


@register_vision_backbone("regnet_x_8gf")
class RegNetX8GFBackbone(RegNet):

    def __init__(self):
        params = RegNetParams.from_init_params(num_layer=23, base_ch=80, ch_progress_slope=49.56, ch_log_step=2.88,
                                               per_group_ch=120, bottleneck_multiplier=1.0, se_ratio=None)
        super().__init__(params)


@register_vision_backbone("regnet_x_16gf")
class RegNetX16GFBackbone(RegNet):

    def __init__(self):
        params = RegNetParams.from_init_params(num_layer=22, base_ch=216, ch_progress_slope=55.59, ch_log_step=2.1,
                                               per_group_ch=128, bottleneck_multiplier=1.0, se_ratio=None)
        super().__init__(params)


@register_vision_backbone("regnet_x_32gf")
class RegNetX32GFBackbone(RegNet):

    def __init__(self):
        params = RegNetParams.from_init_params(num_layer=23, base_ch=320, ch_progress_slope=69.86, ch_log_step=2.0,
                                               per_group_ch=168, bottleneck_multiplier=1.0, se_ratio=None)
        super().__init__(params)


@register_vision_backbone("regnet_y_8gf")
class RegNetY8GFBackbone(RegNet):

    def __init__(self):
        params = RegNetParams.from_init_params(num_layer=17, base_ch=192, ch_progress_slope=76.82, ch_log_step=2.19,
                                               per_group_ch=56, bottleneck_multiplier=1.0, se_ratio=0.25)
        super().__init__(params)


@register_vision_backbone("regnet_y_16gf")
class RegNetY16GFBackbone(RegNet):

    def __init__(self):
        params = RegNetParams.from_init_params(num_layer=18, base_ch=200, ch_progress_slope=106.23, ch_log_step=2.48,
                                               per_group_ch=112, bottleneck_multiplier=1.0, se_ratio=0.25)
        super().__init__(params)


@register_vision_backbone("regnet_y_32gf")
class RegNetY32GFBackbone(RegNet):

    def __init__(self):
        params = RegNetParams.from_init_params(num_layer=20, base_ch=232, ch_progress_slope=115.89, ch_log_step=2.53,
                                               per_group_ch=232, bottleneck_multiplier=1.0, se_ratio=0.25)
        super().__init__(params)


@register_vision_backbone("regnet_y_128gf")
class RegNetY128GFBackbone(RegNet):

    def __init__(self):
        params = RegNetParams.from_init_params(num_layer=27, base_ch=456, ch_progress_slope=160.83, ch_log_step=2.52,
                                               per_group_ch=264, bottleneck_multiplier=1.0, se_ratio=0.25)
        super().__init__(params)
