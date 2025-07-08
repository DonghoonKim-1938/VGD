import torch.nn as nn
from torch.utils.data.dataset import Dataset

DATASET_REGISTRY = {}

__all__ = [
    "register_dataset", "build_dataset",
    "DatasetBase",
]


class DatasetBase(Dataset):

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError


def register_dataset(name: str):
    """Decorator to register dataset."""

    def register_dataset_cls(cls: nn.Module):
        if name in DATASET_REGISTRY:
            raise ValueError(f"[ERROR] Cannot register duplicated dataset {name}.")
        DATASET_REGISTRY[name] = cls
        return cls

    return register_dataset_cls


def build_dataset(name: str, *dataset_args, **dataset_kwargs) -> DatasetBase:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"[ERROR] dataset {name} is not implemented.")

    dataset = DATASET_REGISTRY[name](*dataset_args, **dataset_kwargs)
    return dataset
