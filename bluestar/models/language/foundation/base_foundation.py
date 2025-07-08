from typing import List
import torch
import torch.nn as nn

LANGUAGE_FOUNDATION_REGISTRY = dict()

__all__ = [
    "register_language_foundation",
    "build_language_foundation",
    "LanguageFoundationBase",
]


class LanguageFoundationBase(nn.Module):

    def __init__(self):
        super().__init__()

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return features[-1]


def register_language_foundation(name: str):
    """Decorator to register vision backbones."""

    def register_language_foundation_cls(cls: nn.Module):
        if name in LANGUAGE_FOUNDATION_REGISTRY:
            raise ValueError(f"[ERROR] Cannot register duplicated language foundation {name}.")
        LANGUAGE_FOUNDATION_REGISTRY[name] = cls
        return cls

    return register_language_foundation_cls


def build_language_foundation(name: str, *model_args, **model_kwargs) -> LanguageFoundationBase:
    if name not in LANGUAGE_FOUNDATION_REGISTRY:
        raise ValueError(f"[ERROR] Langauge foundation {name} is not implemented.")

    model = LANGUAGE_FOUNDATION_REGISTRY[name](*model_args, **model_kwargs)
    return model
