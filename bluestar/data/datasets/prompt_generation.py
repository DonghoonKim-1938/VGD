import random
from collections import defaultdict
import json
import os
import unicodedata
from typing import Dict, List

import cv2
from bluestar.data.datasets import LmdbReader
from bluestar.data.datasets.base_dataset import *

from transformers import LlamaTokenizer, DataCollatorWithPadding

@register_dataset("prompt_generation")
class PromptGeneration(DatasetBase):

    def __init__(self, data_path: str, tokenizer:LlamaTokenizer):
        super().__init__()
        self.class_and_superclass = json.load(open(data_path))
        self.initializer_tokens = json.load(open('/home/mjbae/codes/ISLAB_VL/examples/text2img/config/imagenet_classes.json'))
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = "[PAD]"
        # self.tokenizer.padding_side = "left"

        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

    def __len__(self):
        return len(self.class_and_superclass)

    def __getitem__(self, idx: int):

        prompt = f"what does the {self.class_and_superclass[idx]['class']} "\
                 f"the type of {self.class_and_superclass[idx]['superclass']} look like? " \
                 f"describe it with the information of its color, shape in detail that we can draw with your description." \
                 f"please start the sentence with 'This {self.initializer_tokens[idx]['superclass']}'. " \
                 f"please answer in 1 sentence and do not mention anything other than information about the appearance." \

                 # f"please start the sentence with 'a photo of {self.class_and_superclass[idx]['class']} a type of {self.class_and_superclass[idx]['superclass']}.'"

        token = self.tokenizer(prompt)

        return {
            # "id": [self.class_and_superclass[idx]['id']],
            # "class": [self.class_and_superclass[idx]['class']],
            # "superclass": [self.class_and_superclass[idx]['superclass']],
            "input_ids": token["input_ids"],
            "attention_mask": token["attention_mask"],
        }