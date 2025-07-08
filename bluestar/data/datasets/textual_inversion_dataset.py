import json

import torch
import numpy as np
import PIL
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from packaging import version
from transformers import CLIPImageProcessor, CLIPProcessor
from bluestar.utils.chat_template import PromptTemplate as PT

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

imagenet_templates_small = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

imagenet_style_templates_small = [
        "a painting in the style of {}",
        "a rendering in the style of {}",
        "a cropped painting in the style of {}",
        "the painting in the style of {}",
        "a clean painting in the style of {}",
        "a dirty painting in the style of {}",
        "a dark painting in the style of {}",
        "a picture in the style of {}",
        "a cool painting in the style of {}",
        "a close-up painting in the style of {}",
        "a bright painting in the style of {}",
        "a cropped painting in the style of {}",
        "a good painting in the style of {}",
        "a close-up painting in the style of {}",
        "a rendition in the style of {}",
        "a nice painting in the style of {}",
        "a small painting in the style of {}",
        "a weird painting in the style of {}",
        "a large painting in the style of {}",
    ]
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

class SDIPCDataset(Dataset):
    def __init__(
        self,
        data_root
    ):
        self.data_root = data_root
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.transform = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i % self.num_images])

        return self.transform(images=image, return_tensors="pt", padding=True)

class VETIMDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        initialization_token=None,
        descriptions=None,
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.initialization_token = initialization_token
        self.description = descriptions
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        example['input_ids'] = 0
        example['pixel_values'] = 0

        return example



class VGPGDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=224,
        interpolation="bicubic",
        flip_p=0.5,
        center_crop=False,
        super_class:str=None,
        class_name:str=None,
        dataset_name:str=None,
        class_id:str=None
    ):
        self.class_name = class_name
        self.super_class = super_class

        if dataset_name == "imagenet":
            class_info = json.load(open("/home/mjbae/codes/ISLAB_VL/examples/prompt_generation/config/imagenet_150.json"))
            list_info = list(filter(lambda item: item['id'] == class_id, class_info))
            self.class_name = list_info[0]['class']
            self.super_class = list_info[0]['superclass']

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        if class_name is not None:
            self.user_prompt =f"Please generate prompt for generating the image of the {self.super_class} {self.class_name}."
        else:
            self.user_prompt = "Please generate prompt for generating the image."

        self.system_prompt = "You are a respectful and honest visual description generator for Stable Diffusion text prompt." \
                             "Please answer in 1 sentence and do not mention anything other than prompt. " \
                             "Do not mention anything other than visual information. " \
                             "Do not repeat the user and system prompt. " \
                             "Do not use verb. " \
                             # "The format is specified as: " \
                             # "object name, " \
                             # "visual description of object (e.g., scale, color, shape, texture), " \
                             # "description of object action, " \
                             # "After generating the prompt, do not mention anything."

        self.model_prompt = " Answer: Sure, here's a prompt for stable diffusion within 77 tokens:\n "

        self.pt = PT(system_prompt=self.system_prompt)
        self.pt.add_user_message(self.user_prompt, return_prompt=False)
        self.prompt_template = self.pt.build_prompt()

        self.prompt_template += self.model_prompt
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i])

        if not image.mode == "RGB":
            image = image.convert("RGB")
        example["text"] = self.prompt_template
        tokenizer_out = self.tokenizer(
            self.prompt_template,
            return_tensors="pt",
        )
        example["input_ids"] = tokenizer_out["input_ids"].squeeze()
        example["attention_mask"] = tokenizer_out["attention_mask"].squeeze()

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (h, w) = (img.shape[0], img.shape[1],)
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)


        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        example["filename"] = self.image_paths[i].split("/")[-1].split(".")[0]
        return example


class VGPGStyleDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=224,
        interpolation="bicubic",
        flip_p=0.5,
        center_crop=False,
        super_class:str=None,
        class_name:str=None,
        dataset_name:str=None,
        class_id:str=None
    ):
        self.class_name = class_name
        self.super_class = super_class

        if dataset_name == "imagenet":
            class_info = json.load(open("/home/dhkim/codes/ISLAB_VL/bluestar/data/datasets/IN1K_class_info.json"))
            list_info = list(filter(lambda item: item['id'] == class_id, class_info))
            self.class_name = list_info[0]['class']
            self.super_class = list_info[0]['superclass']

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        if class_name is not None:
            self.user_prompt =f"Please generate prompt for generating the image of the {self.super_class} {self.class_name}."
        else:
            self.user_prompt = "Please generate prompt for generating the image."

        self.system_prompt = "You are a respectful and honest visual style description generator for Stable Diffusion text prompt." \
                             "Please answer in 1 sentence and do not mention anything other than prompt. " \
                             "Do not mention anything other than visual information. " \
                             "Do not repeat the user and system prompt. " \
                             "Do not use verb. " \
                             "Use comma for separating the concept. " \
                             "The format is specified as: " \
                             "style name, " \
                             "detailed visual description of style, " \
                             "After generating the prompt, do not mention anything."

        self.model_prompt = " Answer: Sure, here's a prompt for stable diffusion within 77 tokens:\n "

        self.pt = PT(system_prompt=self.system_prompt)
        self.pt.add_user_message(self.user_prompt, return_prompt=False)
        self.prompt_template = self.pt.build_prompt()

        self.prompt_template += self.model_prompt
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i])

        if not image.mode == "RGB":
            image = image.convert("RGB")
        example["text"] = self.prompt_template
        tokenizer_out = self.tokenizer(
            self.prompt_template,
            return_tensors="pt",
        )
        example["input_ids"] = tokenizer_out["input_ids"].squeeze()
        example["attention_mask"] = tokenizer_out["attention_mask"].squeeze()

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (h, w) = (img.shape[0], img.shape[1],)
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)


        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        example["filename"] = self.image_paths[i].split("/")[-1].split(".")[0]
        return example