from typing import List
import torch
import torch.nn as nn

EMBEDDING_REGISTRY = dict()

__all__ = [
    "register_embedding",
    "build_embedding",
]

def register_embedding(name: str):
    """Decorator to register embeddings."""

    def register_embedding_cls(cls: nn.Module):
        if name in EMBEDDING_REGISTRY:
            raise ValueError(f"[ERROR] Cannot register duplicated embedding {name}.")
        EMBEDDING_REGISTRY[name] = cls
        return cls

    return register_embedding_cls


def build_embedding(name: str, *model_args, **model_kwargs) :
    if name not in EMBEDDING_REGISTRY:
        raise ValueError(f"[ERROR] Embedding {name} is not implemented.")

    model = EMBEDDING_REGISTRY[name](*model_args, **model_kwargs)
    return model
