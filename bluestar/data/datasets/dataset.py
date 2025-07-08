from typing import Callable, Optional, Tuple
from PIL import Image
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset

__all__ = ["OurData", "OURDATA_MEAN", "OURDATA_STD"]

OURDATA_MEAN = (0.485, 0.456, 0.406)
OURDATA_STD = (0.229, 0.224, 0.225)


@register_dataset("OurData")
class OurData(DatasetBase):
    """Wrapper of torchvision ImageNet."""

    def __init__(
            self,
            data_dir: str,  # ImageNet root folder, contains 'train' and 'val' folders inside.
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        # ---------------------------------------------------------------- #
        # wrapping torchvision dataset with ours, only using the folder loading part.
        imagenet = ImageFolder(data_dir + "train" if train else data_dir + "val")
        self.data = [i[0] for i in imagenet.samples]  # list of str (file paths)git
        self.targets = [i[1] for i in imagenet.samples]  # list of int (labels)

        self.mean = OURDATA_MEAN
        self.std = OURDATA_STD

        self.transform = transform

    @staticmethod
    def loader(path: str):
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
                return np.array(img)
        except FileNotFoundError as e:
            print(f"[ERROR:DATA] ImageNet file not found: {path}")
            raise e

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_path, label = self.data[index], self.targets[index]
        img = self.loader(data_path)

        if self.transform is not None:
            transformed_data = self.transform(image=img, target=label)
            img, label = transformed_data["image"], transformed_data["target"]

        return img, label, torch.tensor(index)
