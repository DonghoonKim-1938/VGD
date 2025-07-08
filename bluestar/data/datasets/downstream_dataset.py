from collections import defaultdict
import glob
import json
import os
import cv2
import torch

from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset

@register_dataset("image_directory_dataset")
class ImageDirectoryDataset(DatasetBase):
    r"""
    A dataset which reads images from any directory. This class is useful to
    run image captioning inference on our models with any arbitrary images.

    Args:
        data_dir: Path to a directory containing images.
        image_transform: List of image transformations, from either
            `albumentations <https://albumentations.readthedocs.io/en/latest/>`_
            or :mod:`virtex.data.transforms`.
    """
    def __init__(
        self,
        data_dir: str,
        transform,
    ):
        self.image_paths = glob.glob(os.path.join(data_dir, "*"))
        self.image_transform = transform
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        # Remove extension from image name to use as image_id.
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        # Open image from path and apply transformation, convert to CHW format.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image)["image"]


        # Return image id as string so collate_fn does not cast to torch.tensor.
        return {
            "image_id": str(image_id),
            "image": image.clone().detach().to(dtype=torch.float)
        }