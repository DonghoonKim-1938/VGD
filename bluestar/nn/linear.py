import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Linear"]


class Linear(nn.Module):
    """Basic linear projection layer.
    This class is written only for example.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 *, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = bias

        weight = torch.zeros(output_dim, input_dim)
        nn.init.xavier_uniform_(weight, gain=1)
        self.weight = nn.Parameter(weight)

        if self.use_bias:
            bias = torch.zeros(output_dim)
            nn.init.zeros_(bias)
            self.bias = nn.Parameter(bias)
            # self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear forward function.
        :param x:   (..., input_dim)
        :return:    (..., output_dim)
        """
        y = F.linear(x, self.weight, self.bias)
        return y

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, bias={self.use_bias}"


if __name__ == '__main__':
    m = Linear(16, 8, bias=True)
    t_in = torch.empty(6, 16)
    t_out = m(t_in)
    print(t_out.shape)
    print(m)
    print(m.weight.shape)
