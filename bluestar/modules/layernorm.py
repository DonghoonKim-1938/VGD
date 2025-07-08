import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class LayerNorm2d(nn.LayerNorm):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm for channel-axis
        :param x:       (batch_size, ch, h, w)
        :return:        (batch_size, ch, h, w)
        """
        if x.ndim != 4:
            raise ValueError(f"LayerNorm2d require 4D input, but got {x.shape}.")

        var, mean = torch.var_mean(x, dim=1, keepdim=True)  # (1, c, 1, 1)
        inv_std = torch.rsqrt(var + self.eps)
        x = (x - mean) * inv_std
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        # another impl.
        # x = x.permute(0, 2, 3, 1)
        # x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # x = x.permute(0, 3, 1, 2)
        return x
