from typing import List
import torch
import torch.nn as nn

__all__ = ["ExtraVisionBackboneLevel"]


class ExtraVisionBackboneLevel(nn.Module):
    """The modules adds more level on top of backbone features.
    """

    def __init__(
            self,
            input_ch: int,
            output_ch: int,
            num_levels: int = 2,
            *, act_layer=nn.ReLU,
    ) -> None:
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.num_levels = num_levels

        self.layers = nn.ModuleList([
            nn.Conv2d(
                output_ch if (i > 0) else input_ch,
                output_ch,
                kernel_size=3, stride=2, padding=1
            ) for i in range(num_levels)
        ])
        self.act = act_layer()

    def forward(self, feat: torch.Tensor) -> List[torch.Tensor]:
        """Add extra levels with stride.
        :param feat:    (batch_size, input_ch, h, w)
        :return:        (batch_size, output_ch, h/2^l, w/2^l) for each level.
        """
        features = []

        for i in range(self.num_levels):
            feat = self.layers[i](feat)
            if i != self.num_levels - 1:
                feat = self.act(feat)  # apply activation fn except the last one.
            features.append(feat)

        return features
