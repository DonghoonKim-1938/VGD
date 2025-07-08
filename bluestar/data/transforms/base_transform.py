from typing import List
import torch

import torch.nn as nn
import albumentations as A

TRANSFORM_REGISTRY = dict()

__all__ = [
    "register_transform",
    "build_transform",
]

def register_transform(name: str):
    """Decorator to register vision backbones."""
    def register_transform_cls(cls: nn.Module):
        if name in TRANSFORM_REGISTRY:
            raise ValueError(f"[ERROR] Cannot register duplicated vision transform {name}.")
        TRANSFORM_REGISTRY[name] = cls
        return cls

    return register_transform_cls


def build_transform(transform_cfgs: dict) -> TRANSFORM_REGISTRY:
    transform_list = []
    mixup = None
    for name in transform_cfgs.keys():
        if name not in TRANSFORM_REGISTRY:
            raise ValueError(f"[ERROR] Vision transform {name} is not implemented.")

        if name == "mix_up":
            mixup = TRANSFORM_REGISTRY[name](**transform_cfgs[name])
            continue

        transform_list.append(TRANSFORM_REGISTRY[name](**transform_cfgs[name]))

    transform = A.Compose(transform_list)

    return transform, mixup