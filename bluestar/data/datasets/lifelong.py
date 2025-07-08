from typing import Callable, Optional, Tuple
from PIL import Image
import torch
import numpy as np
from torchvision.datasets import ImageFolder

from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset

__all__ = ["LifeLong", "IMAGENET_MEAN", "IMAGENET_STD"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@register_dataset("LifeLong")
class LifeLong(DatasetBase):
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
        split = "/train" if train else "/val"
        real_imagenet = ImageFolder(data_dir + split + "/real")
        crawl_imagenet = ImageFolder(data_dir + split + "/crawl")

        temp = real_imagenet.samples + crawl_imagenet.samples
        temp = list(zip(*temp))
        self.data = list(temp[0])
        self.targets = list(temp[1])
        self.crawled = [0.0] * len(real_imagenet.samples)\
                       + [1.0] * len(crawl_imagenet.samples)
        self.crawled_exist = True

        if sum(self.crawled) == 0.0:
            self.crawled_exist = False

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

    def __getitem__(self, index):
        filename, label = self.data[index], self.targets[index]
        img = self.loader(filename)

        if self.transform is not None:
            transformed_data = self.transform(image=img, target=label)
            img, label = transformed_data["image"], transformed_data["target"]

        if self.crawled_exist:
            crawl = torch.tensor(self.crawled[index])
        else:
            crawl = None

        return img, label, crawl

    @staticmethod
    def fast_collate_imagenet(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        # fast because Z-normalization is applied to batched tensor.
        images = [img[0] for img in batch]  # list of (3, h, w)
        targets = torch.tensor([target[1] for target in batch], dtype=torch.long)

        tensor = torch.stack(images, dim=0).contiguous()  # (b, 3, h, w)
        # mean = torch.as_tensor(list(IMAGENET_MEAN), dtype=torch.float32).view(1, 3, 1, 1)
        # std = torch.as_tensor(list(IMAGENET_STD), dtype=torch.float32).view(1, 3, 1, 1)
        # tensor.sub_(mean).div_(std)
        return tensor, targets