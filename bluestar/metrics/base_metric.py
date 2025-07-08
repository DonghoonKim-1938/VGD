import torch
import torch.nn as nn

METRIC_REGISTRY = {}

__all__ = [
    "register_metric", "build_metric",
    "MetricBase",
]


class MetricBase(nn.Module):

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


def register_metric(name: str):
    """Decorator to register metric."""

    def register_metric_cls(cls: nn.Module):
        if name in METRIC_REGISTRY:
            raise ValueError(f"[ERROR] Cannot register duplicated metric {name}.")
        METRIC_REGISTRY[name] = cls
        return cls

    return register_metric_cls


def build_metric(name: str, *metric_args, **metric_kwargs) -> MetricBase:
    if name not in METRIC_REGISTRY:
        raise ValueError(f"[ERROR] metric {name} is not implemented.")

    metric = METRIC_REGISTRY[name](*metric_args, **metric_kwargs)
    return metric
