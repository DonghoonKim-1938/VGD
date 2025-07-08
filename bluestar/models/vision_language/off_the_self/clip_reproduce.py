
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from bluestar.models.language.embedding import build_embedding
from bluestar.models.vision.backbones.vit import *
from bluestar.models.language.foundation.transformer import *
from bluestar.models.vision.backbones.base_backbone import *
from bluestar.models.language.foundation.base_foundation import *

__all__ = [
    # Vision Transformer attention block
    "CLIPVision",
    "CLIPLanguage",
]

class QuickGELU(torch.nn.Module):
   def forward(self, x: torch.Tensor):
       return x * torch.sigmoid(1.702 * x)

class CLIPVision(ViTBase):
    def __init__(
            self,
            **kwargs
    ):
        kwargs.update(
            act_layer=QuickGELU
        )
        super().__init__(**kwargs)
        self.patch_embed.proj.bias = None
        self.pre_norm = nn.LayerNorm(self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(self.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        self.absolute_pos_embed = nn.Parameter(torch.zeros(self.patch_embed.num_patches + 1, self.embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward_features(self, x):
        B, C, H, W = x.shape
        cls_tokens = self.cls_token.reshape(1,1,-1).repeat(B, 1, 1)  # same as expand(B, -1, -1)
        x = self.patch_embed(x)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.pre_norm(x)

        features = []
        features.append(x)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        x = self.norm(x)  # B L C
        features.append(x[:, 0])  # Only CLS feature

        return features

class CLIPLanguage(TransformerEncoder):
    def __init__(
            self,
            **kwargs
    ):
        kwargs.update(
            act_layer=QuickGELU
        )
        embed_cfg = kwargs.pop("embedding")
        super().__init__(**kwargs)
        self.embedding = build_embedding(**embed_cfg)
        self.embed_dim = kwargs['embed_dim']
        self.ln_final = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lookaheadmask = make_mask(x, self.embedding.padding_idx)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, lookaheadmask,)
        x = self.ln_final(x)
        return x




_model_scale_dict = {
    'vision_transformer_b16': {
        'embed_dim': 768 , 'patch_size': 16, 'depths': 12, 'num_heads': 12, 'drop_rate': 0.0, 'drop_path_rate':0.1, 'patch_norm': False
    },
    'vision_transformer_l14': {
        'embed_dim': 1024, 'patch_size': 16, 'depths': 24, 'num_heads': 16, 'drop_rate': 0.1, 'drop_path_rate':0.2, 'patch_norm': False
    },
    'vision_transformer_l14_336': {
        'embed_dim': 1024, 'img_size':336, 'patch_size': 14, 'depths': 24, 'num_heads': 16, 'drop_rate': 0.1, 'drop_path_rate':0.2, 'patch_norm': False
    },
}


@register_vision_backbone("clip_vision_transformer_b16")
class ViTB16Backbone(CLIPVision):
    def __init__(self):
        super().__init__(**_model_scale_dict['vision_transformer_b16'])


@register_vision_backbone("clip_vision_transformer_l14")
class ViTL14Backbone(CLIPVision):
    def __init__(self):
        super().__init__(**_model_scale_dict['vision_transformer_l14'])

@register_vision_backbone("clip_vision_transformer_l14_336")
class ViTL14336Backbone(CLIPVision):
    def __init__(self):
        super().__init__(**_model_scale_dict['vision_transformer_l14_336'])


@register_language_foundation("clip_transformer_encoder")
class TransformerEncoderBackbone(CLIPLanguage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
