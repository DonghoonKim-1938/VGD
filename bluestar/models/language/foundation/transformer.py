import torch
import torch.nn as nn
import copy
from bluestar.modules.multihead_attention import MultiHeadAttention
from timm.models.layers import Mlp
from .base_foundation import register_language_foundation


__all__ = [
    # Transformer block
    "EncoderBlock",
    "DecoderBlock",

    # Transformer Enc & Dec Block
    "TransformerEncoder",
    "TransformerDecoder",

    "Transformer",
    'make_mask'

]

class EncoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 norm_first=False,
                 ):
        super().__init__()
        self.attn = MultiHeadAttention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop_rate,
            proj_drop=drop_rate,
            self_attention=True,
        )

        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop_rate
        )

        self.norm_first=norm_first
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.dropout1 = nn.Dropout(p=drop_rate)
        self.dropout2 = nn.Dropout(p=drop_rate)


    def forward(self, enc_input, mask = None,):

        x = enc_input
        if self.norm_first:
            x = x + self.dropout1(self.attn(self.norm1(x), mask=mask))
            x = x + self.dropout2(self.mlp(self.norm2(x)))

        else:
            x = self.norm1(x + self.dropout1(self.attn(x, mask=mask)))
            x = self.norm2(x + self.dropout2(self.mlp(x)))

        return x


@register_language_foundation("transformer_encoder")
class TransformerEncoder(nn.Module):

    def __init__(
            self,
            embed_dim,
            num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.1,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            norm_first=False,
            num_layers=1,
            norm=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_first=norm_first,
            ) for _ in range(num_layers)
        )
        self.num_layers=num_layers
        self.norm=norm
    def forward(
            self,
            enc_input,
            mask,
    ) -> torch.Tensor:
        x = enc_input

        for layer in self.layers:
            x = layer(x, mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 norm_first = False,
                 ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop_rate,
            proj_drop=drop_rate,
            self_attention=True,
        )

        self.cross_attn = MultiHeadAttention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop_rate,
            proj_drop=drop_rate,
            self_attention=False # cross attention
        )

        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop_rate
        )

        self.norm_first=norm_first
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.norm3 = norm_layer(embed_dim)
        self.dropout1 = nn.Dropout(p=drop_rate)
        self.dropout2 = nn.Dropout(p=drop_rate)
        self.dropout3 = nn.Dropout(p=drop_rate)


    def forward(self, dec_input, enc_output, self_attn_mask, cross_attn_mask,):

        x = dec_input
        if self.norm_first:
            x = x + self.dropout1(self.self_attn(self.norm1(x), mask=self_attn_mask))
            x = x + self.dropout2(self.cross_attn(self.norm2(x), y=enc_output, mask=cross_attn_mask))
            x = x + self.dropout3(self.mlp(self.norm3(x)))

        else: # default
            x = self.norm1(x + self.dropout1(self.self_attn(x, mask=self_attn_mask)))
            x = self.norm2(x + self.dropout2(self.cross_attn(x, y= enc_output, mask= cross_attn_mask)))
            x = self.norm3(x + self.dropout3(self.mlp(x)))

        return x

@register_language_foundation("transformer_decoder")
class TransformerDecoder(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.1,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            norm_first=False,
            device=None,
            dtype=None,
            num_layers=1,
            norm=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_first=norm_first,
            ) for _ in range(num_layers))
        self.num_layers=num_layers
        self.norm=norm

    def forward(
            self,
            dec_input,
            enc_output,
            self_attn_mask,
            cross_attn_mask,
    ):
        x = dec_input

        for layer in self.layers:
            x = layer(x, enc_output, self_attn_mask, cross_attn_mask,)

        if self.norm is not None:
            x = self.norm(x)

        return x


class Transformer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_head,
                 num_enc_layers,
                 num_dec_layers,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 norm_first=False,
                 ):
        super().__init__()
        encoder_block = EncoderBlock(
            embed_dim,
            num_head,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            act_layer,
            norm_layer,
            norm_first,
        )
        encoder_norm = norm_layer(embed_dim)
        self.encoder = TransformerEncoder(
            encoder_block,
            num_enc_layers,
            encoder_norm,
        )

        decoder_block = DecoderBlock(
            embed_dim,
            num_head,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            act_layer,
            norm_layer,
            norm_first,
        )
        decoder_norm = norm_layer(embed_dim)
        self.decoder = TransformerDecoder(
            decoder_block,
            num_dec_layers,
            decoder_norm,
        )

        self._reset_parameters()
        self.embed_dim = embed_dim
        self.num_head = num_head


    def forward(self, enc_input, dec_input, enc_mask=None, dec_self_attn_mask=None, dec_cross_attn_mask=None):

        enc_output = self.encoder(enc_input, enc_mask)
        output = self.decoder(dec_input, enc_output, dec_self_attn_mask, dec_cross_attn_mask)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

def make_mask(x:torch.Tensor, pad_idx:int, y:torch.Tensor=None,):
    # x.shape: [b, s] : before embedding
    pad_mask = (x != pad_idx).unsqueeze(-1).unsqueeze(1) # torch.Size([b, 1, s, 1])

    if y is not None: # cross attention
        sub_mask = torch.ones(x.shape[-1], y.shape[1]).type(torch.ByteTensor).to(x.device)

    else: # self attention
        sub_mask = torch.tril(torch.ones(x.shape[-1], x.shape[-1])).type(torch.ByteTensor).to(x.device)

    return pad_mask & sub_mask


_model_scale_dict = {
    'transformer_base': {
        'embed_dim': 1024, 'max_seq_length': 1024, 'num_layers': 24, 'num_heads': 16, 'drop_rate': 0.1,
        'drop_path_rate': 0.2
    },
    'transformer_large': {
        'embed_dim': 1024, 'max_seq_length': 1024, 'num_layers': 24, 'num_heads': 16, 'drop_rate': 0.1,
        'drop_path_rate': 0.2
    },
    'transformer_encoder': {
        'embed_dim': 1024, 'max_seq_length': 1024, 'num_layers': 24, 'num_heads': 16, 'drop_rate': 0.1,
        'drop_path_rate': 0.2
    },
    'transformer_decoder': {
        'embed_dim': 1024, 'max_seq_length': 1024, 'num_layers': 24, 'num_heads': 16, 'drop_rate': 0.1,
        'drop_path_rate': 0.2
    },

}


@register_language_foundation("transformer_base")
class TransformerBase(Transformer):
    def __init__(self):
        super().__init__(**_model_scale_dict['transformer_base'])

@register_language_foundation("transformer_large")
class TransformerLarge(Transformer):
    def __init__(self):
        super().__init__(**_model_scale_dict['transformer_large'])
