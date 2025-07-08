
import torch
import torch.nn as nn

from bluestar.modules.drop_path import DropPath
from bluestar.modules.multihead_attention import MultiHeadAttention

from timm.models.layers import PatchEmbed, Mlp, trunc_normal_

from .base_backbone import VisionBackboneBase, register_vision_backbone

__all__ = [
    # Vision Transformer attention block
    "ViTBlock",
    # Vision Transformer base model
    "ViTBase",
    # Vision Transformer models
    "ViTB16Backbone",
    "ViTL16Backbone",
    "ViTH14Backbone",
    # DINO model
    "DINOB16Backbone"
]


class ViTBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        # self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):


        shortcut = x
        x = self.norm1(x) # Pre-norm
        x = self.attn(x)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class ViTBase(VisionBackboneBase):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
            self,
            embed_dim=768,
            depths=12,
            num_heads=12,
            img_size=224,
            patch_size=16,
            patch_norm=False,
            in_chans=3,               # use only default value
            mlp_ratio=4.,             # use only default value
            qkv_bias=True,            # use only default value
            drop_rate=0.,             # use only default value
            attn_drop_rate=0.0,       # use only default value
            drop_path_rate=0.1,       # use only default value
            norm_layer=nn.LayerNorm,  # use only default value
            act_layer=nn.GELU,
            **kwargs
    ):
        super().__init__()

        self.num_layers = depths
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        self._features_ch.append(self.embed_dim)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)  # 0.02???

        # absolute position embedding
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList([
            ViTBlock(
                dim=self.embed_dim,
                input_resolution=(self.patch_grid[0], self.patch_grid[1]),
                num_heads=num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer = act_layer,
        ) for i in range(self.num_layers)
        ])

        self.norm = norm_layer(self.embed_dim)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B, C, H, W = x.shape
        cls_tokens = self.cls_token.repeat(B, 1, 1)  # same as expand(B, -1, -1)
        x = self.patch_embed(x)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        features = []
        features.append(x)

        for layer in self.layers:
          x = layer(x)
          features.append(x)

        x = self.norm(x)  # B L C
        features.append(x[:,0]) # Only CLS feature

        return features

_model_scale_dict = {
    'vision_transformer_b16': {
        'embed_dim': 768 , 'patch_size': 16, 'depths': 12, 'num_heads': 12, 'drop_rate': 0.0, 'drop_path_rate':0.1, 'patch_norm': True
    },
    'vision_transformer_l16': {
        'embed_dim': 1024, 'patch_size': 16, 'depths': 24, 'num_heads': 16, 'drop_rate': 0.1, 'drop_path_rate':0.2, 'patch_norm': True
    },
    'vision_transformer_h14': {
        'embed_dim': 1280, 'patch_size': 14, 'depths': 32, 'num_heads': 16, 'drop_rate': 0.1, 'drop_path_rate':0.2, 'patch_norm': True
    },
    'dino_b16': {
        'embed_dim': 768 , 'patch_size': 16, 'depths': 12, 'num_heads': 12, 'drop_rate': 0.0, 'drop_path_rate':0.1, 'patch_norm': False
    },
}


@register_vision_backbone("vision_transformer_b16")
class ViTB16Backbone(ViTBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['vision_transformer_b16'])


@register_vision_backbone("vision_transformer_l16")
class ViTL16Backbone(ViTBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['vision_transformer_l16'])


@register_vision_backbone("vision_transformer_h14")
class ViTH14Backbone(ViTBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['vision_transformer_h14'])


@register_vision_backbone("dino_b16")
class DINOB16Backbone(ViTBase):

    def __init__(self):
        super().__init__(**_model_scale_dict['dino_b16'])