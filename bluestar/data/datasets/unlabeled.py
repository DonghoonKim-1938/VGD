from typing import Callable, Optional, Tuple
from PIL import Image
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset

__all__ = ["UnlabeledDataset", "IMAGENET_MEAN", "IMAGENET_STD"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@register_dataset("UnlabeledDataset")
class UnlabeledDataset(DatasetBase):
    """Dataset for Unlabeled Raw Images (Inference)"""

    def __init__(
            self,
            data_dir: str,
            train: bool = True,
            transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        split = "/train" if train else "/val"
        imagenet = ImageFolder(data_dir + split + "/crawl")
        self.data = [i[0] for i in imagenet.samples]

        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.data[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        index = torch.tensor([index])

        return index, img