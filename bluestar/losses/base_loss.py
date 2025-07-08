import torch
import torch.nn as nn

LOSS_REGISTRY = {}

__all__ = [
    "register_loss", "build_loss",
    "LossBase",
]


class LossBase(nn.Module):

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


def register_loss(name: str):
    """Decorator to register loss."""

    def register_loss_cls(cls: nn.Module):
        if name in LOSS_REGISTRY:
            raise ValueError(f"[ERROR] Cannot register duplicated loss {name}.")
        LOSS_REGISTRY[name] = cls
        return cls

    return register_loss_cls


def build_loss(name: str, *loss_args, **loss_kwargs) -> LossBase:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"[ERROR] Loss {name} is not implemented.")

    loss = LOSS_REGISTRY[name](*loss_args, **loss_kwargs)
    return loss
