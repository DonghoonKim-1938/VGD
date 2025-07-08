from typing import List
import torch
import torch.nn as nn

VISION_NECK_REGISTRY = {}

__all__ = [
    "register_vision_neck", "build_vision_neck",
    "VisionNeckBase",
]


class VisionNeckBase(nn.Module):

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError


def register_vision_neck(name: str):
    """Decorator to register vision neck."""

    def register_vision_neck_cls(cls: nn.Module):
        if name in VISION_NECK_REGISTRY:
            raise ValueError(f"[ERROR] Cannot register duplicated vision neck {name}.")
        VISION_NECK_REGISTRY[name] = cls
        return cls

    return register_vision_neck_cls


def build_vision_neck(name: str, *model_args, **model_kwargs) -> VisionNeckBase:
    if name not in VISION_NECK_REGISTRY:
        raise ValueError(f"[ERROR] Vision neck {name} is not implemented.")

    model = VISION_NECK_REGISTRY[name](*model_args, **model_kwargs)
    return model
