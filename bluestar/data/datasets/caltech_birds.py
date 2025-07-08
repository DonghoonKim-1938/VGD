import json
from typing import Callable, Optional, Tuple
from PIL import Image
import torch
import numpy as np
from torchvision.datasets import ImageFolder

from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset

__all__ = ["CaltechBirds"]

@register_dataset("CaltechBirds")
class CaltechBirds(DatasetBase):
    """Wrapper of torchvision CaltechBirds."""

    def __init__(self,
            data_dir: str,
            split:str="train",
            transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        # ---------------------------------------------------------------- #
        # wrapping torchvision dataset with ours, only using the folder loading part.
        # imagenet = ImageFolder(data_dir + "/train" + "/crawl" if train else data_dir + "/val" + "/crawl") # for 2012
        self.data_dir = data_dir
        # data_path = data_dir + "CUB_200_2011/images"
        # caltech_birds = ImageFolder(data_path)

        # imagenet = ImageFolder(data_dir + "/train" if train else "/val" ) else 2012
        # data = [i[0] for i in caltech_birds.samples]  # list of str (file paths)
        # targets = [i[1] for i in caltech_birds.samples]  # list of int (labels)

        self.transform = transform
        self.split = split
        self.split_file = json.load(open(data_dir + f"/{split}.json"))
        self.data = [*self.split_file.keys()]
        self.targets = [*self.split_file.values()]
        self.classes = json.load(open(data_dir + "classes.json"))

    @staticmethod
    def loader(path: str):
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
                # return img
                return np.array(img)
        except FileNotFoundError as e:
            print(f"[ERROR:DATA] ImageNet file not found: {path}")
            raise e

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self.data[index], self.targets[index]
        img = self.data_dir + img.replace('caltech-birds','')
        img = self.loader(img)

        if self.transform is not None:
            transformed_data = self.transform(image=img, target=label)
            img, label = transformed_data["image"], transformed_data["target"]

        return img, label