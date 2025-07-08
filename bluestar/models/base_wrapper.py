from typing import Iterator, Tuple
import torch
import torch.nn as nn

from bluestar.models.base_model import ModelBase

__all__ = ["WrapperBase"]


class WrapperBase(ModelBase):
    """Wrapper to keep model and its behaviors together."""

    def __init__(self, model: ModelBase):
        super().__init__()
        self.model = model

    def state_dict(self, destination=None, prefix: str = ""):  # noqa
        return self.model.state_dict(destination, prefix=prefix)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    """
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        return self.model.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        return self.model.named_buffers(prefix=prefix, recurse=recurse)

    def children(self) -> Iterator[nn.Module]:
        return self.model.children()

    def named_children(self) -> Iterator[Tuple[str, nn.Module]]:
        return self.model.named_children()

    def modules(self) -> Iterator[nn.Module]:
        return self.model.modules()

    def named_modules(self, prefix: str = '', remove_duplicate: bool = True):  # noqa
        return self.model.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)

    def get_parameter(self, target: str) -> nn.Parameter:
        return self.model.get_parameter(target)

    def get_buffer(self, target: str) -> torch.Tensor:
        return self.model.get_buffer(target)

    def get_submodule(self, target: str) -> nn.Module:
        return self.model.get_submodule(target)
    """
