import random
import time
from typing import Callable, Dict, List

import numpy as np
import torch

from .coco_captions import CocoCaptionsDataset
from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset
from torchvision.utils import save_image



@register_dataset("CaptioningDataset")
class CaptioningDataset(DatasetBase):
    r"""
    A dataset which provides image-caption (forward and backward) pairs from
    a COCO Captions annotation file. This is used for pretraining tasks which
    use captions - bicaptioning, forward captioning and token classification.

    Args:
        data_root: Path to dataset directory containing images and annotations.
        split: Name of COCO 2017 split to read. One of ``{"train", "val"}``.
        tokenizer: Tokenizer which maps word tokens to their integer IDs.
        image_transform: List of image transformations, from either
            `albumentations <https://albumentations.readthedocs.io/en/latest/>`_
            or :mod:`virtex.data.transforms`.
        max_caption_length: Maximum number of tokens to keep in caption tokens.
            Extra tokens will be trimmed from the right end of the token list.
    """

    def __init__(
        self,
        data_dir: str,
        train: bool,
        tokenizer,
        transform,
        max_caption_length: int = 30,
        lmdb: bool = False,
    ):
        self.data_dir = data_dir
        self._dset = CocoCaptionsDataset(data_dir, train, lmdb)
        self.tokenizer = tokenizer
        self.image_transform = transform
        self.max_caption_length = max_caption_length

        # Short handles for common tokens for convenience:
        self.padding_idx = tokenizer.token_to_id("<unk>") #0
        self.sos_id = tokenizer.token_to_id("[SOS]") #1
        self.eos_id = tokenizer.token_to_id("[EOS]") #2


    def __len__(self):
        return len(self._dset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        # keys: {"image_id", "image", "captions"}
        instance = self._dset[idx]
        image_id, image, captions = (
            instance["image_id"],
            instance["image"],
            instance["captions"],
        )
        caption = random.choice(captions)


        # Transform image-caption pair and convert image from HWC to CHW format.
        # Pass in caption to image_transform due to paired horizontal flip.
        # Caption won't be tokenized/processed here.

        image_caption = self.image_transform(image=image, caption=caption)
        image, caption = image_caption["image"], image_caption["caption"]

        caption_tokens = [self.sos_id, *self.tokenizer.encode(caption), self.eos_id]
        caption_tokens = caption_tokens[: self.max_caption_length]

        return {
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "image": image.clone().detach().to(dtype=torch.float),
            "caption_tokens": torch.tensor(caption_tokens, dtype=torch.long),
            # "noitpac_tokens": torch.tensor(caption_tokens, dtype=torch.long).flip(0),
            "caption_lengths": torch.tensor(len(caption_tokens), dtype=torch.long),
        }

    def collate_fn(
        self, data: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        # Pad `caption_tokens` and `masked_labels` up to this length.
        caption_tokens = torch.nn.utils.rnn.pad_sequence(
            [d["caption_tokens"] for d in data],
            batch_first=True,
            padding_value=self.padding_idx,
        )

        return {
            "image_id": torch.stack([d["image_id"] for d in data], dim=0),
            "image": torch.stack([d["image"] for d in data], dim=0),
            "caption_tokens": caption_tokens,
            #"noitpac_tokens": noitpac_tokens,
            "caption_lengths": torch.stack([d["caption_lengths"] for d in data]),
        }
