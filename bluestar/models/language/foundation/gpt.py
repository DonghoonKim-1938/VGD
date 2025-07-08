from .transformer import TransformerDecoder
from .base_foundation import register_language_foundation


__all__ = [
    # GPT models
    "GPTBase",
    "GPTLarge"

]


_model_scale_dict = {
    'gpt_base': {
        'embed_dim': 768 , 'max_seq_length': 512, 'depths': 12, 'num_heads': 12, 'drop_rate': 0.1, 'drop_path_rate':0.1
    },


    'gpt_large': {
        'embed_dim': 1280, 'max_seq_length': 1024, 'depths': 36, 'num_heads': 20, 'drop_rate': 0.1, 'drop_path_rate':0.2
    },
}


@register_language_foundation("gpt_base")
class GPTBase(TransformerDecoder):
    def __init__(self):
        super().__init__(**_model_scale_dict['gpt_base'])

@register_language_foundation("gpt_large")
class GPTLarge(TransformerDecoder):
    def __init__(self):
        super().__init__(**_model_scale_dict['gpt_large'])
