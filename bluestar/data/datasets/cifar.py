import os.path
import pickle
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import copy

from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset

@register_dataset("cifar10")
class CIFAR10(DatasetBase):

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root=root
        self.transform=transform
        self.target_transform=target_transform
        self.split = split

        if self.split == "train":
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name) # cifar-100-python/train or cifar-100-python/test
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        self.templates = [
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ]


    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])

        if self.base_folder == "cifar-10-batches-py": # cifar10
            self.classes = [
                'airplane',
                'automobile',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck',
            ]
        elif self.base_folder == "cifar-100-python": # cifar100
            self.classes = [
                'apple',
                'aquarium fish',
                'baby',
                'bear',
                'beaver',
                'bed',
                'bee',
                'beetle',
                'bicycle',
                'bottle',
                'bowl',
                'boy',
                'bridge',
                'bus',
                'butterfly',
                'camel',
                'can',
                'castle',
                'caterpillar',
                'cattle',
                'chair',
                'chimpanzee',
                'clock',
                'cloud',
                'cockroach',
                'couch',
                'crab',
                'crocodile',
                'cup',
                'dinosaur',
                'dolphin',
                'elephant',
                'flatfish',
                'forest',
                'fox',
                'girl',
                'hamster',
                'house',
                'kangaroo',
                'keyboard',
                'lamp',
                'lawn mower',
                'leopard',
                'lion',
                'lizard',
                'lobster',
                'man',
                'maple tree',
                'motorcycle',
                'mountain',
                'mouse',
                'mushroom',
                'oak tree',
                'orange',
                'orchid',
                'otter',
                'palm tree',
                'pear',
                'pickup truck',
                'pine tree',
                'plain',
                'plate',
                'poppy',
                'porcupine',
                'possum',
                'rabbit',
                'raccoon',
                'ray',
                'road',
                'rocket',
                'rose',
                'sea',
                'seal',
                'shark',
                'shrew',
                'skunk',
                'skyscraper',
                'snail',
                'snake',
                'spider',
                'squirrel',
                'streetcar',
                'sunflower',
                'sweet pepper',
                'table',
                'tank',
                'telephone',
                'television',
                'tiger',
                'tractor',
                'train',
                'trout',
                'tulip',
                'turtle',
                'wardrobe',
                'whale',
                'willow tree',
                'wolf',
                'woman',
                'worm',
            ] # this is only for CLIP

        # with open(path, "rb") as infile:
        #     data = pickle.load(infile, encoding="latin1")
        #     self.classes = data[self.meta["key"]] # original code

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image, label = self.data[index], self.targets[index]
        # image: ndarray, shape: 32,32,3

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # image = Image.fromarray(image) # TODO check

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.target_transform is not None:
            raise NotImplementedError
            # label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.data)

@register_dataset("cifar100")
class CIFAR100(CIFAR10):

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

