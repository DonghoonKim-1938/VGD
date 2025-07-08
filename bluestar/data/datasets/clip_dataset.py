import random
import torch

from typing import Dict, List
from bluestar.data.datasets.base_dataset import DatasetBase


class CLIPDataset(DatasetBase):
    def __init__(
        self,
        tokenizer,
        transform,
        d_set,
        max_caption_length: int = 77,
        single_caption: bool = True,
    ):
        self._dset = d_set
        self.tokenizer = tokenizer
        self.image_transform = transform
        self.max_caption_length = max_caption_length

        self.sos_id = tokenizer.encoder["<|startoftext|>"]
        self.eos_id = tokenizer.encoder["<|endoftext|>"]
        self.padding_idx = 0
        self.padding_caption = [self.padding_idx for i in range(self.max_caption_length)]

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
        image_captions = self.image_transform(image=image, caption=captions)

        image, captions = image_captions["image"], image_captions["caption"]

        caption_tokens = []
        caption_lengths = []
        if len(captions) != 5:
            skip_idx = random.choice(range(len(captions)))
            captions = [*captions[:skip_idx], *captions[skip_idx+1:]]
        if len(captions) != 5:
            skip_idx = random.choice(range(len(captions)))
            captions = [*captions[:skip_idx], *captions[skip_idx + 1:]]

        for caption in captions:
            caption_token = [self.sos_id, *self.tokenizer.encode('a photo of '+caption), self.eos_id, ]
            caption_token = caption_token[: self.max_caption_length]
            caption_lengths.append(len(caption_token))
            caption_token = [*caption_token, *self.padding_caption][: self.max_caption_length]
            caption_tokens.append(caption_token)

        return {
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "image": image.clone().detach().to(dtype=torch.float),
            "caption_tokens": torch.tensor(caption_tokens, dtype=torch.long),
            "caption_lengths": torch.tensor(caption_lengths, dtype=torch.long),
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
            "caption_lengths": torch.stack([d["caption_lengths"] for d in data]),
        }
