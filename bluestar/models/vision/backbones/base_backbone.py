from typing import List
import torch
import torch.nn as nn

VISION_BACKBONE_REGISTRY = dict()

__all__ = [
    "register_vision_backbone",
    "build_vision_backbone",
    "VisionBackboneBase",
]


class VisionBackboneBase(nn.Module):

    def __init__(self):
        super().__init__()
        self._features_ch = list()

    @property
    def features_ch(self) -> List[int]:
        return self._features_ch

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return features[-1]


def register_vision_backbone(name: str):
    """Decorator to register vision backbones."""

    def register_vision_backbone_cls(cls: nn.Module):
        if name in VISION_BACKBONE_REGISTRY:
            raise ValueError(f"[ERROR] Cannot register duplicated vision backbone {name}.")
        VISION_BACKBONE_REGISTRY[name] = cls
        return cls

    return register_vision_backbone_cls


def build_vision_backbone(name: str, *model_args, **model_kwargs) -> VisionBackboneBase:
    if name not in VISION_BACKBONE_REGISTRY:
        raise ValueError(f"[ERROR] Vision backbone {name} is not implemented.")

    model = VISION_BACKBONE_REGISTRY[name](*model_args, **model_kwargs)
    return model
