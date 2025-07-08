import torch
from torch import nn

class space_squeeze(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.squeeze_ratio = ratio
        self.squeeze_ratio_sq = ratio * ratio

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # shape of x
        b, c, h, w = x.shape

        # shape of squeezed x
        squeezed_h = h // self.squeeze_ratio
        squeezed_w = w // self.squeeze_ratio
        squeezed_c = c * self.squeeze_ratio_sq

        # reference block position
        h0_idx, w0_idx = torch.meshgrid(
            torch.arange(0, h, self.squeeze_ratio), torch.arange(0, w, self.squeeze_ratio)
        )

        # relative block indices to be stacked up
        del_h, del_w = torch.meshgrid(
            torch.arange(0, self.squeeze_ratio), torch.arange(0, self.squeeze_ratio)
        )
        del_h = del_h.flatten()
        del_w = del_w.flatten()

        # grouping the blocks and stack it up to channel dimension
        o = [x[...,h0_idx + d_h, w0_idx + d_w] for d_h, d_w in zip(del_h, del_w)]
        o = torch.cat(o,dim=1)

        return o.contiguous()

    def inverse(self, z:torch.Tensor) -> torch.Tensor:
        # shape of z
        b, c, h, w = z.shape

        # shape of squeezed z
        unsqueezed_h = h * self.squeeze_ratio
        unsqueezed_w = w * self.squeeze_ratio
        unsqueezed_c = c // self.squeeze_ratio_sq

        # reference block position
        c0_idx = torch.arange(0,c,unsqueezed_c)

        # grouping the blocks and reshape it to the original shape
        o = [z[...,c_i:(c_i+unsqueezed_c),:,:].unsqueeze(-1) for c_i in c0_idx]
        o = torch.cat(o, dim=-1)
        o = o.reshape(b,unsqueezed_c, h, w, self.squeeze_ratio, self.squeeze_ratio)
        o = o.transpose(-2, -3)
        o = o.reshape(b,unsqueezed_c,unsqueezed_h,unsqueezed_w)

        return o.contiguous()