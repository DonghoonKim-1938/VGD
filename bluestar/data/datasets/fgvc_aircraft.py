import json
from typing import Callable, Optional, Tuple
from PIL import Image
import torch
import numpy as np
from torchvision.datasets import FGVCAircraft

from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset

__all__ = ["FGVCAircraft"]

@register_dataset("FGVCAircraft")
class FGVCAircraft(DatasetBase):
    """Wrapper of torchvision FGVCAircraft."""

    def __init__(
            self,
            data_dir: str,
            split: str = "train", # train/trainval/val/test
            transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        # ---------------------------------------------------------------- #
        # wrapping torchvision dataset with ours, only using the folder loading part.
        # imagenet = ImageFolder(data_dir + "/train" + "/crawl" if train else data_dir + "/val" + "/crawl") # for 2012
        self.data_dir = data_dir
        # self.data_path = data_dir + "/fgvc_aircraft/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/"
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
            print(f"[ERROR:DATA] Dataset file not found: {path}")
            raise e

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self.data[index], self.targets[index]
        img = self.data_dir + img
        # img = self.data_dir + img.replace('fgvc_aircraft','')
        img = self.loader(img)

        if self.transform is not None:
            transformed_data = self.transform(image=img, target=label)
            img, label = transformed_data["image"], transformed_data["target"]

        return img, label