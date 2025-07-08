import math
import timm.data
import cv2
import torch
import random
import numpy as np

import albumentations as A
from typing import Iterable, Tuple, Optional, List, Sequence, Union

from .base_transform import register_transform
from albumentations import KeypointType
from bluestar.utils.transform_utils import to_tuple, is_rgb_image, is_grayscale_image
import torch.nn.functional as F
import warnings
__all__ = [
    # pytorch
    "RandomResizedCrop",
    "RandomHorizontalFlip",
    "ToTensor",
    "Resize",
    "CenterCrop",
    "RandomErasing",
    "MixUp",
    "SmallestResize",
    "Normalize",
    "RandomBrightnessContrast",
    "HueSaturationValue",
    "ToGray",
    "RandomCrop",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


interpolations = {
    'nearest' :  cv2.INTER_NEAREST,
    'bicubic' :  cv2.INTER_CUBIC,
    'bilinear':  cv2.INTER_LINEAR
}


@register_transform("random_resized_crop")
class RandomResizedCrop(A.RandomResizedCrop):
    def __init__(
            self,
            img_size,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation:str = 'bicubic',
            always_apply=False,
            p=1.0,
    ):
        interpolation =  interpolations[interpolation]
        super().__init__(img_size, img_size, tuple(scale), tuple(ratio), interpolation, always_apply, p)


@register_transform("random_horizontal_flip")
class RandomHorizontalFlip(A.BasicTransform):

    def __init__(self,
                 always_apply: bool = False,
                 p: float = 0.5):
        super().__init__(always_apply,p)

    @property
    def targets(self):
        return {"image": self.apply, "caption": self.apply_to_caption}

    def apply(self, img, **params):
        return cv2.flip(img, 1)

    def apply_to_caption(self, captions, **params):
        if type(captions) is str:
            captions = (
            captions.replace("left", "[TMP]")
            .replace("right", "left")
            .replace("[TMP]", "right")
        )
            return captions
        else:
            cp = []
            for caption in captions:
                caption = (
                    caption.replace("left", "[TMP]")
                    .replace("right", "left")
                    .replace("[TMP]", "right")
                )
                cp.append(caption)
            return cp


@register_transform("color_jitter")
class ColorJitter(A.ColorJitter):

    def __init__(self,
                 brightness,
                 contrast,
                 saturation,
                 hue,
                 always_apply=False,
                 p=0.8,):
        super().__init__(brightness, contrast, saturation, hue, always_apply,p,)

@register_transform("smallest_resize")
class SmallestResize(A.SmallestMaxSize):

    def __init__(self, img_size, interpolation:str = 'bicubic', always_apply=False, p=1):
        interpolation =  interpolations[interpolation]
        super().__init__(img_size, interpolation, always_apply, p)

@register_transform("resize")
class Resize(A.Resize):

    def __init__(self, size, interpolation:str = 'bicubic', always_apply=False, p=1):
        interpolation =  interpolations[interpolation]
        super().__init__(size, size, interpolation, always_apply, p)

@register_transform("random_crop")
class RandomCrop(A.RandomCrop):

    def __init__(self, size, always_apply=False, p=1):
        super().__init__(size, size,  always_apply, p)

@register_transform("center_crop")
class CenterCrop(A.CenterCrop):

    def __init__(self, img_size, always_apply=False, p=1.0):
        super().__init__(img_size, img_size, always_apply, p)


@register_transform("mix_up")
class MixUp(timm.data.Mixup):

    def __init__(
            self,
            mixup_alpha=1.,
            cutmix_alpha=0.,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            correct_lam=True,
            label_smoothing=0.1,
            num_classes=1000
    ):

        super().__init__(
            mixup_alpha,
            cutmix_alpha,
            cutmix_minmax,
            prob,
            switch_prob,
            mode,
            correct_lam,
            label_smoothing,
            num_classes
        )


@register_transform("normalize")
class Normalize(A.Normalize):

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD, **cfg):
        super().__init__(mean, std)


@register_transform("to_tensor")
class ToTensor(A.BasicTransform):
    # The function is the implementation of totensor of torchvision.
    # This will convert image and mask to torch.Tensor.
    # The imput image 'HWC' will converted to pytorch 'CHW' tensor.

    def __init__(self, transpose_mask=False, **cfg):
        super().__init__(always_apply=True, p=1.0)

        # Check if the mask contains 3 dim.
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks
        }

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose((2, 0, 1)))

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose((2, 0, 1))
        return torch.from_numpy(mask)

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}


@register_transform("random_erasing")
class RandomErasing(A.DualTransform):

    def __init__(
            self,
            max_count  : Optional[int] = None,
            min_count  : int = 1,
            max_area   : float = 1/3,
            min_area   : float = 0.02,
            max_aspect : Optional[float] = None,
            min_aspect : float = 0.3,
            mode       : str = 'constant', # can be "random" or "constant"
            mask_fill_value: Optional[int] = None,
            always_apply: bool = False,
            p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        self.max_count = max_count if max_count is not None else min_count
        self.min_count = min_count
        self.max_area = max_area
        self.min_area = min_area
        self.max_aspect = max_aspect if max_aspect is not None else min_aspect
        self.min_aspect = min_aspect
        self.mode = mode
        self.fill_value = 0 if mode == 'constant' else None
        self.mask_fill_value = mask_fill_value

        self.log_aspect_ratio = (math.log(self.min_aspect), math.log(self.max_aspect))

        if not 0 < self.min_count <= self.max_count:
            raise ValueError("Invalid combination of min_holes and max_holes. Got: {}".format([min_count, max_count]))

        self.check_range(self.max_area)
        self.check_range(self.min_area)
        self.check_range(self.max_aspect)
        self.check_range(self.min_aspect)

        if not 0 < self.min_area <= self.max_area:
            raise ValueError(
                "Invalid combination of min_area and max_area. Got: {}".format([min_area, max_area])
            )
        if not 0 < self.min_aspect <= self.max_aspect:
            raise ValueError("Invalid combination of min_aspect and max_aspect. Got: {}".format([min_aspect, max_aspect]))

    def check_range(self, dimension):
        if isinstance(dimension, float) and not 0 <= dimension < 1.0:
            raise ValueError(
                "Invalid value {}. If using floats, the value should be in the range [0.0, 1.0)".format(dimension)
            )

    def cutout(
            self,
            img: np.ndarray,
            holes: Iterable[Tuple[int, int, int, int]],
            value_for_holes: Union[int, float, Iterable[np.ndarray]] = 0
    ) -> np.ndarray:
        # Make a copy of the input image since we don't want to modify it directly
        img = img.copy()
        for (x1, y1, x2, y2), value in zip(holes,value_for_holes):
            img[y1:y2, x1:x2] = value
        return img

    def apply(
            self,
            img: np.ndarray,
            value_for_holes: Union[int, float, np.ndarray] = 0,
            holes: Iterable[Tuple[int, int, int, int]] = (),
            **params
    ) -> np.ndarray:
        return self.cutout(img, holes, value_for_holes)

    def apply_to_mask(
            self,
            img: np.ndarray,
            mask_fill_value: Union[int, float] = 0,
            holes: Iterable[Tuple[int, int, int, int]] = (),
            **params
    ) -> np.ndarray:
        if mask_fill_value is None:
            return img
        return self.cutout(img, holes, mask_fill_value)

    def get_params_dependent_on_targets(self, params):

        img = params["image"]
        height, width = img.shape[:2]
        area = height * width

        holes = []
        value_for_holes = []
        for _n in range(random.randint(self.min_count, self.max_count)):

            if all(
                    [
                        isinstance(self.min_area, float),
                        isinstance(self.min_aspect, float),
                        isinstance(self.max_area, float),
                        isinstance(self.max_aspect, float),
                    ]
            ):

                for attempt in range(10):

                    hole_area    = (area * random.uniform(self.min_area, self.max_area))
                    hole_aspect  = math.exp(random.uniform(*self.log_aspect_ratio))

                    hole_height = int(round(math.sqrt(hole_area * hole_aspect)))
                    hole_width  = int(round(math.sqrt(hole_area / hole_aspect)))

                    if hole_width < width and hole_height < height:

                        y1 = random.randint(0, height - hole_height)
                        x1 = random.randint(0, width - hole_width)

                        y2 = y1 + hole_height
                        x2 = x1 + hole_width

                        holes.append((x1, y1, x2, y2))
                        if self.mode == "random":
                            value_for_holes.append(
                                np.random.randn(hole_height,hole_width,3)
                            )
                        else:
                            value_for_holes = self.fill_value

                        break

            else:
                raise ValueError(
                    "Min area, max area, \
                    min aspect and max aspect \
                    should all either be ints or floats. \
                    Got: {} respectively".format(
                        [
                            type(self.min_area),
                            type(self.min_aspect),
                            type(self.max_area),
                            type(self.max_aspect),
                        ]
                    )
                )

        return {"holes": holes, "value_for_holes": value_for_holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def _keypoint_in_hole(self, keypoint: KeypointType, hole: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = hole
        x, y = keypoint[:2]
        return x1 <= x < x2 and y1 <= y < y2

    def apply_to_keypoints(
            self, keypoints: Sequence[KeypointType], holes: Iterable[Tuple[int, int, int, int]] = (), **params
    ) -> List[KeypointType]:
        result = []
        for hole in holes:
            remaining_keypoints = []
            for kp in keypoints:
                if not self._keypoint_in_hole(kp, hole):
                    remaining_keypoints.append(kp)
            result = remaining_keypoints
        return result

    def get_transform_init_args_names(self):
        return (
            "max_count",
            "max_area",
            "max_aspect",
            "min_count",
            "min_area",
            "min_aspect",
            "mask_fill_value",
        )


@register_transform("random_brightness_contrast")
class RandomBrightnessContrast(A.RandomBrightnessContrast):
    def __init__(
            self,
            brightness_limit=0.2,
            contrast_limit=0.2,
            brightness_by_max=True,
            always_apply=False,
            p=0.5,
    ):
        super(RandomBrightnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max

    def apply(self, img, alpha=1.0, beta=0.0, **params):
        return F.brightness_contrast_adjust(img, alpha, beta, self.brightness_by_max)

    def get_params(self):
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def get_transform_init_args_names(self):
        return ("brightness_limit", "contrast_limit", "brightness_by_max")

@register_transform("huesaturationvalue")
class HueSaturationValue(A.HueSaturationValue):
    def __init__(
            self,
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            always_apply=False,
            p=0.5,
    ):
        super(HueSaturationValue, self).__init__(always_apply, p)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0, **params):
        if not is_rgb_image(image) and not is_grayscale_image(image):
            raise TypeError("HueSaturationValue transformation expects 1-channel or 3-channel images.")
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self):
        return {
            "hue_shift": random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
            "sat_shift": random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
            "val_shift": random.uniform(self.val_shift_limit[0], self.val_shift_limit[1]),
        }

    def get_transform_init_args_names(self):
        return ("hue_shift_limit", "sat_shift_limit", "val_shift_limit")

@register_transform("to_gray")
class ToGray(A.ToGray):

    def apply(self, img, **params):
        if is_grayscale_image(img):
            warnings.warn("The image is already gray.")
            return img
        if not is_rgb_image(img):
            raise TypeError("ToGray transformation expects 3-channel images.")

        return F.to_gray(img)

    def get_transform_init_args_names(self):
        return ()
