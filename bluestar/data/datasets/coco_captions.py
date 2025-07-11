import random
from collections import defaultdict
import json
import os
import unicodedata
from typing import Dict, List

import cv2
from bluestar.data.datasets import LmdbReader
from bluestar.data.datasets.base_dataset import *

@register_dataset("coco_captions")
class CocoCaptionsDataset(DatasetBase):
    r"""
    A PyTorch dataset to read COCO Captions dataset and provide it completely
    unprocessed. This dataset is used by various task-specific datasets
    in :mod:`~virtex.data.datasets` modules.

    Args:
        data_root: Path to the COCO dataset root directory.
        split: Name of COCO 2017 split to read. One of ``{"train", "val"}``.
    """

    def __init__(self, data_dir: str, train: bool, lmdb: bool = False):
        self.lmdb = lmdb

        if train:
            split = "train"
        else:
            split = "val"

        if not lmdb:
            # Get paths to image directory and annotation file.
            image_dir = os.path.join(data_dir, f"images/{split}2017")
            captions = json.load(
                open(os.path.join(data_dir, "annotations", f"captions_{split}2017.json"))
            )
            # Collect list of captions for each image.
            captions_per_image: Dict[int, List[str]] = defaultdict(list)

            for ann in captions["annotations"]:
                # Perform common normalization (lowercase, trim spaces, NKFC strip
                # accents and NKFC normalization).
                caption = ann["caption"].lower()
                caption = unicodedata.normalize("NFKD", caption)
                caption = "".join([chr for chr in caption if not unicodedata.combining(chr)])

                captions_per_image[ann["image_id"]].append(caption)

            # Collect image file for each image (by its ID).
            image_filepaths: Dict[int, str] = {
                im["id"]: os.path.join(image_dir, im["file_name"])
                for im in captions["images"]
            }
            # Keep all annotations in memory. Make a list of tuples, each tuple
            # is ``(image_id, file_path, list[captions])``.
            self.instances = [
                (im_id, image_filepaths[im_id], captions_per_image[im_id])
                for im_id in captions_per_image.keys()
            ]
        else:
            image_dir = os.path.join(data_dir, f"lmdb/{split}.lmdb")
            self.instances = LmdbReader(image_dir)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int):
        if not self.lmdb:
            image_id, image_path, captions = self.instances[idx]
            # image shape ndarray,: (height, width, channels), dtype: uint8
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        else:
            image_id, image, captions = self.instances[f"{idx}"].values()

        return {"image_id": image_id, "image": image, "captions": captions}
