from .transformer import TransformerEncoder
from .base_foundation import register_language_foundation


__all__ = [
    # BERT models
    "BertBase",
    "BertLarge"

]

# TODO: BERTBase
# TODO:Base, Large -> Registry Decorator


_model_scale_dict = {
    'bert_base': {
        'embed_dim': 768 , 'max_seq_length': 512, 'depths': 12, 'num_heads': 12, 'drop_rate': 0.1, 'drop_path_rate':0.1
    },
    'bert_large': {
        'embed_dim': 1024, 'max_seq_length': 512, 'depths': 24, 'num_heads': 16, 'drop_rate': 0.1, 'drop_path_rate':0.2
    },
}


@register_language_foundation("bert_base")
class BertBase(TransformerEncoder):
    def __init__(self):
        super().__init__(**_model_scale_dict['bert_base'])

@register_language_foundation("bert_large")
class BertLarge(TransformerEncoder):
    def __init__(self):
        super().__init__(**_model_scale_dict['bert_large'])
