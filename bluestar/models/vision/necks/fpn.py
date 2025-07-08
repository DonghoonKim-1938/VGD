from typing import List
import torch
import torch.nn as nn

from bluestar.modules.convbn import ConvBN2d
from .base_neck import VisionNeckBase, register_vision_neck


@register_vision_neck("fpn")
class FPNNeck(VisionNeckBase):
    """Feature pyramid network"""

    def __init__(
            self,
            features_ch: List[int],  # input feature channels.
            ch: int,  # output feature channel.
            *, upsample_type="bilinear",
            use_norm: bool = False,
    ) -> None:
        super().__init__()
        self.features_ch = features_ch
        self.ch = ch

        lateral_layers = []
        output_layers = []

        lateral_kwargs = dict(kernel_size=1, stride=1, padding=0)
        conv_kwargs = dict(kernel_size=3, stride=1, padding=1)

        # we assume that feature resolution decreases by halves (high-to-low resolution)
        for i, feat_ch in enumerate(features_ch):
            if use_norm:
                lateral_conv = ConvBN2d(feat_ch, ch, **lateral_kwargs)
                output_conv = ConvBN2d(ch, ch, **conv_kwargs)
            else:
                lateral_conv = nn.Conv2d(feat_ch, ch, **lateral_kwargs)
                output_conv = nn.Conv2d(ch, ch, **conv_kwargs)
            lateral_layers.append(lateral_conv)
            output_layers.append(output_conv)

        # order is low-to-high resolution
        self.lateral_layers = nn.ModuleList(list(reversed(lateral_layers)))
        self.output_layers = nn.ModuleList(list(reversed(output_layers)))

        if upsample_type == "bilinear":
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        elif upsample_type == "nearest":
            self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        else:
            raise ValueError(f"Unsupported upsample type {upsample_type}.")

    def _check_channels(self, features: List[torch.Tensor]) -> None:
        if len(features) != len(self.features_ch):
            raise ValueError(f"FPN got invalid number of features: {len(features)} vs. {len(self.features_ch)}.")
        for feat, ch in zip(features, self.features_ch):
            if feat.shape[1] != ch:
                raise ValueError(f"FPN got invalid channels: {tuple(feat.shape)} vs. {ch}.")

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """FPN forward.
        :param features:        (batch_size, ch_l, h_l, w_l) for each level, high-to-low resolution.
        :return:    features    (batch_size, ch, h_l, w_l) for each level, high-to-low resolution.
        """
        self._check_channels(features)

        results = []

        # lowest resolution (top of pyramid)
        feat = self.lateral_layers[0](features[-1])
        results.append(self.output_layers[0](feat))

        for i in range(1, len(features)):
            backbone_feat = self.lateral_layers[i](features[-i - 1])
            upsample_feat = self.upsample(feat)
            feat = backbone_feat + upsample_feat
            results.insert(0, self.output_layers[i](feat))  # put at front

        return results
