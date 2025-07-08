
from torch.nn.init import trunc_normal_
from bluestar.models.vision.backbones.vit import *
from bluestar.models.vision.backbones.base_backbone import *

import math
import torch
import torch.nn as nn

from functools import reduce
from operator import mul
from torch.nn import Dropout

__all__ = [
    # Vision Transformer attention block
    "VPTBase",
    "VPTDEEPBase",
    "VPTb16",
    "VPTDEEPb16",
]

class VPTBase(ViTBase):
    def __init__(
            self,
            **kwargs
    ):
        self.prompt_num = kwargs["prompt_num"][kwargs['data_name']]
        super().__init__(**kwargs)
        self.prompt_dropout = Dropout(kwargs['prompt_dropout'])
        self.cls_token = nn.Parameter(torch.zeros(self.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        self.absolute_pos_embed = nn.Parameter(torch.zeros(self.patch_embed.num_patches + 1, self.embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)

        self.val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_num , self.embed_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -self.val, self.val)

    def embeding_with_prompt(self, x):
        B, C, H, W = x.shape
        cls_tokens = self.cls_token.reshape(1, 1, -1).repeat(B, 1, 1)  # same as expand(B, -1, -1)
        x = self.patch_embed(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = torch.cat((x[:, :1, :], self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1)), x[:, 1:, :]), dim=1)

        return x
    def forward_features(self, x):
        x = self.embeding_with_prompt(x)

        features = []
        features.append(x)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        x = self.norm(x)
        features.append(x[:, 0])  # Only CLS feature

        return features


class VPTDEEPBase(VPTBase):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        total_d_layer = self.num_layers - 1
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(total_d_layer, self.prompt_num, self.embed_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.deep_prompt_embeddings.data, -self.val, self.val)

    def forward_features(self, x):
        x = self.embeding_with_prompt(x)

        features = []
        features.append(x)

        for i in range(self.num_layers):
            if i==0:
                x = self.layers[0](x)
            else:
                deep_prompt_emb = self.prompt_dropout(self.deep_prompt_embeddings[i - 1].expand(x.shape[0], -1, -1))
                x = torch.cat((x[:,:1,:], deep_prompt_emb, x[:,1+self.prompt_num:,:]), dim=1)
                x = self.layers[i](x)
            features.append(x)

        x = self.norm(x)
        features.append(x[:, 0])  # Only CLS feature

        return features


_model_scale_dict = {
    'vision_transformer_b16': {
        'embed_dim': 768 , 'patch_size': 16, 'depths': 12, 'num_heads': 12, 'drop_rate': 0.1, 'drop_path_rate':0.1, 'patch_norm': False
    },
    'vision_transformer_l14': {
        'embed_dim': 1024, 'patch_size': 16, 'depths': 24, 'num_heads': 16, 'drop_rate': 0.1, 'drop_path_rate':0.2, 'patch_norm': False
    },
}

@register_vision_backbone("vpt_vision_transformer_b16")
class VPTb16(VPTBase):
    def __init__(self, **kwargs):
        _model_scale_dict['vision_transformer_b16'].update(kwargs)
        super().__init__(**_model_scale_dict['vision_transformer_b16'])

@register_vision_backbone("vpt_deep_vision_transformer_b16")
class VPTDEEPb16(VPTDEEPBase):
    def __init__(self, **kwargs):
        _model_scale_dict['vision_transformer_b16'].update(kwargs)
        super().__init__(**_model_scale_dict['vision_transformer_b16'])

