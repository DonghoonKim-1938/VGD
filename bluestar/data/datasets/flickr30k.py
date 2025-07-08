import random
from collections import defaultdict
import json
import os
import unicodedata
from typing import Dict, List

import cv2
from bluestar.data.datasets import LmdbReader
from bluestar.data.datasets.base_dataset import *

@register_dataset("flickr30k")
class Flickr30kDataset(DatasetBase):
    """
    Args:
        data_root: Path to the flickr30k dataset root directory.
    """
    def __init__(self, data_dir: str, train: bool, lmdb: bool = False):
        self.lmdb = lmdb
        #
        # if train:
        #     split = "train"
        # else:
        #     split = "val"

        if not lmdb:
            # Get paths to image directory and annotation file.
            image_dir = os.path.join(data_dir, f"Images/")
            captions = json.load(
                open(os.path.join(data_dir, "flickr30k_captions.json"))
            )
            # Collect list of captions for each image.
            captions_per_image: Dict[int, List[str]] = defaultdict(list)

            for ann in captions:
                # Perform common normalization (lowercase, trim spaces, NKFC strip
                # accents and NKFC normalization).
                caption = ann["comment"].lower()
                caption = unicodedata.normalize("NFKD", caption)
                caption = "".join([chr for chr in caption if not unicodedata.combining(chr)])

                captions_per_image[int(ann["image_name"].replace(".jpg", ""))].append(caption)

            # Collect image file for each image (by its ID).
            image_filepaths: Dict[int, str] = {
                int(ann["image_name"].replace(".jpg", "")): os.path.join(image_dir, ann["image_name"])
                for ann in captions
            }

            # Keep all annotations in memory. Make a list of tuples, each tuple
            # is ``(image_id, file_path, list[captions])``.
            self.instances = [
                (im_id, image_filepaths[im_id], captions_per_image[im_id])
                for im_id in captions_per_image.keys()
            ]

        else:
            image_dir = os.path.join(data_dir, f"lmdb/flickr30k.lmdb")
            self.instances = LmdbReader(image_dir)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        if not self.lmdb:
            image_id, image_path, captions = self.instances[idx]
            # shape: (height, width, channels), dtype: uint8
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        else:
            image_id, image, captions = self.instances[f"{idx}"].values()

        return {"image_id": image_id, "image": image, "captions": captions}
