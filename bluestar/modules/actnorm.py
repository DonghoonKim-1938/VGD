import torch
import torch.nn as nn

__all__ = ["ActNorm2d"]


class ActNorm(nn.Module):

    def __init__(
            self,
            num_channel: int,
            dim: int = 1,  # channel dim
            eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_channel = num_channel
        self.dim = dim
        self.eps = eps

        self.log_scale = nn.Parameter(torch.zeros(num_channel))
        self.shift = nn.Parameter(torch.zeros(num_channel))
        self.register_buffer("_initialized", torch.tensor(False))

    def reset_(self, flag: bool = False):
        self._initialized = torch.tensor(flag)  # noqa
        return self

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)  # check the input dimension
        if x.shape[self.dim] != self.num_channel:
            raise ValueError(f"ActNorm channel dimension mismatch: {x.shape[self.dim]} vs. {self.num_channel}.")

        _shape = [1] * x.ndim  # [1, 1, 1, 1]
        _shape[self.dim] = self.num_channel  # [1, c, 1, 1]

        if not self._initialized:
            # initialize parameters with the first mini-batch mean and std

            _dims = list(range(x.ndim))  # [0, 1, 2, 3]
            _dims.pop(self.dim)  # [0, 2, 3]
            with torch.no_grad():
                var, mean = torch.var_mean(x, dim=_dims)  # (c,)
                inv_std = torch.rsqrt(var + self.eps)
                self.log_scale.data = torch.log(inv_std)  # log(1 / std)
                self.shift.data = -mean * inv_std  # -mean / std
                # y = x * exp(log_scale) + shift = x / std - mean / std = (x - mean) / std

            self.reset_(flag=True)

        scale = torch.exp(self.log_scale).view(_shape)
        bias = self.shift.view(_shape)
        x = x * scale + bias
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        _shape = [1] * x.ndim  # [1, 1, 1, 1]
        _shape[self.dim] = self.num_channel  # [1, c, 1, 1]

        neg_scale = torch.exp(-self.log_scale).view(_shape)
        bias = self.shift.view(_shape)

        x = (x - bias) * neg_scale
        return x


class ActNorm2d(ActNorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"ActNorm2d expected 4D tensor but got {x.shape} input.")
