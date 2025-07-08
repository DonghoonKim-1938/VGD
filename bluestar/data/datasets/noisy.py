import torch
import json
import numpy as np
import torch.nn.functional as F

from PIL import Image
from typing import Callable, Optional, Tuple

from torchvision.datasets import ImageFolder
from bluestar.data.datasets import LmdbReader
from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset

__all__ = ["Noisy", "IMAGENET_MEAN", "IMAGENET_STD"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@register_dataset("Noisy")
class Noisy(DatasetBase):
    """Wrapper of torchvision ImageNet."""

    def __init__(
            self,
            labeled_data_dir: str,  # ImageNet root folder, contains 'train' and 'val' folders inside.
            unlabeled_data_dir: str,  # ImageNet root folder, contains 'train' and 'val' folders inside.
            train: bool = True,
            transform: Optional[Callable] = None,
            num_classes: int = 1000,
    ) -> None:
        super().__init__()
        # ---------------------------------------------------------------- #
        # wrapping torchvision dataset with ours, only using the folder loading part.
        part_dir = '/train' if train else '/val'
        labeled_data = ImageFolder(labeled_data_dir + part_dir + "/real") # IN1K-2012

        self.data = [i[0] for i in labeled_data.samples]
        self.labeled_targets = [i[1] for i in labeled_data.samples]  # list of int (labels)
        self.num_classes = num_classes

        # labeled: 1.0
        self.labeled = [1.0] * len(self.data)

        # unlabeled data
        # soft-label from machine-filtering process
        model_pred_file = unlabeled_data_dir + '/lmdb/model_pred.lmdb'
        model_filter_id = unlabeled_data_dir + '/lmdb/machine_filtered_idx.json'

        # model prediction lmdb
        unlabeled_filtered_id = np.concatenate(json.load(open(model_filter_id))).tolist()
        self.data += unlabeled_filtered_id
        self.unlabeled_data = LmdbReader(model_pred_file) # pred for entire crawled img

        # unlabeled: 0.0
        self.labeled += [0.0] * (len(self.data) - len(self.labeled))

        # transform of the image
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

        labeled = self.labeled[index]

        if labeled == 1.0: # labeled
            path, label = self.data[index], self.labeled_targets[index]
            label = torch.tensor(label)
            label = F.one_hot(label, self.num_classes) + 0.0
            img = self.loader(path)

        elif labeled == 0.0:
            data_info = self.unlabeled_data.get_data(f"{int(self.data[index])}")
            path, label = data_info["filename"], data_info["p_dist"]
            label = torch.tensor(label, dtype=torch.float)
            img = self.loader(path)

        else:
            ValueError(f"labeled (value: {labeled}) should be 0.0 or 1.0")

        if self.transform is not None:
            transformed_data = self.transform(image=img, target=label)
            img, label = transformed_data["image"], transformed_data["target"]

        return img, label, torch.tensor([labeled], dtype=torch.float)
