# This is random augmentation implementation for albumentation.
# The baseline of the code is from timm.data.auto_augment

import re
from typing import Union, List

import albumentations as A
import numpy as np
from PIL import Image
from .base_transform import register_transform
from .transforms import IMAGENET_MEAN, IMAGENET_STD
from timm.data.transforms import str_to_pil_interp
from timm.data.auto_augment import (
    rand_augment_ops,
    _LEVEL_DENOM,
    _RAND_TRANSFORMS,
    _RAND_INCREASING_TRANSFORMS,
)


__all__ = [
    "RandAugment",
]

_RAND_CHOICE_WEIGHTS_0 = {
    "Rotate": 0.3,
    "ShearX": 0.2,
    "ShearY": 0.2,
    "TranslateXRel": 0.1,
    "TranslateYRel": 0.1,
    "Color": 0.025,
    "Sharpness": 0.025,
    "AutoContrast": 0.025,
    "Solarize": 0.005,
    "SolarizeAdd": 0.005,
    "Contrast": 0.005,
    "Brightness": 0.005,
    "Equalize": 0.005,
    "Posterize": 0,
    "Invert": 0,
}

def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs



@register_transform("rand_augment")
class RandAugment(A.ImageOnlyTransform):
    def __init__(self, config_str, img_size: Union[int, List], interpolation="bicubic", mean=IMAGENET_MEAN):

        super(RandAugment, self).__init__(always_apply=True, p=1.0)

        img_size_min = min(img_size) if isinstance(img_size, list) else img_size

        hparams = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
            interpolation=str_to_pil_interp(interpolation)
        )

        ra_ops, num_layers, choice_weights = self.rand_augment_transform(config_str, hparams)

        self.ops = ra_ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def apply(
            self,
            img: np.ndarray,
            **params
    ):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights
        )

        img = Image.fromarray(img)

        for op in ops:
            img = op(img)

        img = np.array(img)
        return img

    def get_transform_init_args_names(self):
        return ("ops", "num_layers", "choice_weights")

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.num_layers}, ops='
        for op in self.ops:
            fs += f'\n\t{op}'
        fs += ')'
        return fs

    def rand_augment_transform(self, config_str, hparams):
        """
        Create a RandAugment transform

        :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
        dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
        sections, not order sepecific determine
            'm' - integer magnitude of rand augment
            'n' - integer num layers (number of transform ops selected per image)
            'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
            'mstd' -  float std deviation of magnitude noise applied, or uniform sampling if infinity (or > 100)
            'mmax' - set upper bound for magnitude to something other than default of  _LEVEL_DENOM (10)
            'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
        Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
        'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2

        :param hparams: Other hparams (kwargs) for the RandAugmentation scheme

        :return: A PyTorch compatible Transform
        """
        magnitude = _LEVEL_DENOM  # default to _LEVEL_DENOM for magnitude (currently 10)
        num_layers = 2  # default to 2 ops per image
        weight_idx = None  # default to no probability weights for op choice
        transforms = _RAND_TRANSFORMS
        config = config_str.split('-')
        assert config[0] == 'rand'
        config = config[1:]
        for c in config:
            cs = re.split(r'(\d.*)', c)
            if len(cs) < 2:
                continue
            key, val = cs[:2]
            if key == 'mstd':
                # noise param / randomization of magnitude values
                mstd = float(val)
                if mstd > 100:
                    # use uniform sampling in 0 to magnitude if mstd is > 100
                    mstd = float('inf')
                hparams.setdefault('magnitude_std', mstd)
            elif key == 'mmax':
                # clip magnitude between [0, mmax] instead of default [0, _LEVEL_DENOM]
                hparams.setdefault('magnitude_max', int(val))
            elif key == 'inc':
                if bool(val):
                    transforms = _RAND_INCREASING_TRANSFORMS
            elif key == 'm':
                magnitude = int(val)
            elif key == 'n':
                num_layers = int(val)
            elif key == 'w':
                weight_idx = int(val)
            else:
                assert False, 'Unknown RandAugment config section'
        ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams, transforms=transforms)
        choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)
        return ra_ops, num_layers, choice_weights

