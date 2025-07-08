import glob
import os

from PIL import Image

from bluestar.data.datasets.base_dataset import *
from transformers import AutoProcessor
from bluestar.utils.chat_template import PromptTemplate as PT
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


@register_dataset("image_captioning")
class ImageCaptioning(DatasetBase):

    def __init__(self, cfg):
        self.image_path = cfg["data_dir"]
        self.images = []
        for img_extension in IMG_EXTENSIONS:
            self.images.extend(glob.glob(os.path.join(self.image_path, ('*' + img_extension))))

        self.prompt = cfg["prompt"]


        self.processor = AutoProcessor.from_pretrained(cfg["model_ckpt"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        inputs = self.processor(text=self.prompt, images=image, return_tensors="pt")
        inputs = {k:inputs[k].squeeze(0) for k in inputs.keys()}
        return [self.images[idx], inputs]