import torch
import torch.nn as nn

VISION_HEAD_REGISTRY = {}

__all__ = [
    "register_vision_head", "build_vision_head",
    "VisionHeadBase",
]


class VisionHeadBase(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def register_vision_head(name: str):
    """Decorator to register vision heads."""

    def register_vision_head_cls(cls: nn.Module):
        if name in VISION_HEAD_REGISTRY:
            raise ValueError(f"[ERROR] Cannot register duplicated vision head {name}.")
        VISION_HEAD_REGISTRY[name] = cls
        return cls

    return register_vision_head_cls


def build_vision_head(name: str, *model_args, **model_kwargs) -> VisionHeadBase:
    if name not in VISION_HEAD_REGISTRY:
        raise ValueError(f"[ERROR] Vision head {name} is not implemented.")

    model = VISION_HEAD_REGISTRY[name](*model_args, **model_kwargs)
    return model
