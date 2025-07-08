import torch

from torch import nn


__all__ = ["MultiHeadAttention"]

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 attn_drop,
                 proj_drop,
                 qkv_bias=True,
                 self_attention:bool=True,
                 ):
        super().__init__()

        self.self_attention = self_attention

        if self_attention:
            # self attention
            self.W_qkv = nn.Linear(dim, dim*3, bias=qkv_bias) # QKV

        else:
            # cross attention
            self.W_q = nn.Linear(dim, dim, bias=qkv_bias)  # dim: 1024
            self.W_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(p=proj_drop)

        self.num_head = num_heads
        self.head_dim = dim // num_heads

        self.scale = float(self.head_dim ** -0.5)

    def forward(self, x:torch.Tensor, y:torch.Tensor = None, mask=None,):

        B, N, C = x.shape

        if y is None:
            qkv = self.W_qkv(x).reshape(B, N, 3, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(dim=0) # B, H, N, C
        else:
            q = self.W_q(x).reshape(B, N, self.num_head, self.head_dim).permute(0, 2, 1, 3)
            kv = self.W_kv(y).reshape(B, y.shape[1], 2, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(dim=0) # B, H, N, C

        q = q * self.scale

        attn = torch.matmul(q, k.transpose(-2,-1))

        # to mask pad token attention
        # mask: (B, 1, seq_len, seq_len)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e4)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
