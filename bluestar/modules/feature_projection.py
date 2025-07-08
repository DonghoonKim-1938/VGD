import torch
import torch.nn as nn

__all__ = ["VisionToLanguage"]


class VisionToLanguage(nn.Module):
    """linear projection of features from vision to language model"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ) -> None:
        super().__init__()
        self.linear_projection = nn.Linear(
            in_channels, out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """changing dimensions + linear projection.
        :param x:       (batch_size, input_ch, input_h, input_w)
        :return:        (batch_size, output_ch, hidden_size(embedding))
        """
        b, c, h, w = x.shape
        x = x.view(b, c, -1) # (b, c, hw)
        x = x.permute(0, 2, 1) # (b, hw, c)
        x = self.linear_projection(x)
        return x
