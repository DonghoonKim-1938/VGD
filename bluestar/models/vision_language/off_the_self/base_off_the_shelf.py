from typing import List
import torch
import torch.nn as nn

OTS_BACKBONE_REGISTRY = dict()

__all__ = [
    "register_ots_backbone",
    "build_ots_backbone",
    "OTSBackboneBase",
]


class OTSBackboneBase(nn.Module):

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


def register_ots_backbone(name: str):
    """Decorator to register ots backbones."""

    def register_ots_backbone_cls(cls: nn.Module):
        if name in OTS_BACKBONE_REGISTRY:
            raise ValueError(f"[ERROR] Cannot register duplicated ots backbone {name}.")
        OTS_BACKBONE_REGISTRY[name] = cls
        return cls

    return register_ots_backbone_cls


def build_ots_backbone(name: str, *model_args, **model_kwargs) -> OTSBackboneBase:
    if name not in OTS_BACKBONE_REGISTRY:
        raise ValueError(f"[ERROR] Ots backbone {name} is not implemented.")

    model = OTS_BACKBONE_REGISTRY[name](*model_args, **model_kwargs)
    return model
