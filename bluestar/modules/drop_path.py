import torch
from torch import nn

# from torchvision.ops.stochastic_depth import StochasticDepth  # equivalent Class

__all__ = ["DropPath"]


class DropPath(nn.Module):

    def __init__(self, drop_prob: float, rescale: bool = True):
        super().__init__()
        if not (0 <= drop_prob < 1):
            raise ValueError(f"DropPath drop_prob should be in [0, 1), but got {drop_prob}.")

        self.drop_prob = drop_prob
        self.rescale = rescale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DropPath forward.
        :param x:   (batch_size, ...)
        :return:    (batch_size, ...)
        """
        if (self.training is False) or (self.drop_prob == 0):
            return x

        batch_size = x.shape[0]
        keep_prob = 1 - self.drop_prob

        mask_shape = [batch_size] + [1] * (x.ndim - 1)
        binary_mask = torch.zeros(*mask_shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)

        if self.rescale:
            x /= keep_prob
        output = x * binary_mask
        return output
