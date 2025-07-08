import torch.nn as nn

DECODER_REGISTRY = dict()

__all__ = [
    "register_decoder",
    "build_decoder",
]

def register_decoder(name: str):
    """Decorator to register vision backbones."""

    def register_decoder_cls(cls: nn.Module):
        if name in DECODER_REGISTRY:
            raise ValueError(f"[ERROR] Cannot register duplicated decoder {name}.")
        DECODER_REGISTRY[name] = cls
        return cls

    return register_decoder_cls


def build_decoder(name: str, *model_args, **model_kwargs):
    if name not in DECODER_REGISTRY:
        raise ValueError(f"[ERROR] Langauge foundation {name} is not implemented.")

    model = DECODER_REGISTRY[name](*model_args, **model_kwargs)
    return model
