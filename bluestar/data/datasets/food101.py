import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import cv2
import PIL.Image

from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset

@register_dataset("Food101")
class Food101(DatasetBase):

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root=root
        self.transform=transform
        self.target_transform=target_transform
        self._split = split
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"


        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())

        self.classess = sorted(metadata.keys())
        self.classes = [
            'apple pie',
            'baby back ribs',
            'baklava',
            'beef carpaccio',
            'beef tartare',
            'beet salad',
            'beignets',
            'bibimbap',
            'bread pudding',
            'breakfast burrito',
            'bruschetta',
            'caesar salad',
            'cannoli',
            'caprese salad',
            'carrot cake',
            'ceviche',
            'cheese plate',
            'cheesecake',
            'chicken curry',
            'chicken quesadilla',
            'chicken wings',
            'chocolate cake',
            'chocolate mousse',
            'churros',
            'clam chowder',
            'club sandwich',
            'crab cakes',
            'creme brulee',
            'croque madame',
            'cup cakes',
            'deviled eggs',
            'donuts',
            'dumplings',
            'edamame',
            'eggs benedict',
            'escargots',
            'falafel',
            'filet mignon',
            'fish and chips',
            'foie gras',
            'french fries',
            'french onion soup',
            'french toast',
            'fried calamari',
            'fried rice',
            'frozen yogurt',
            'garlic bread',
            'gnocchi',
            'greek salad',
            'grilled cheese sandwich',
            'grilled salmon',
            'guacamole',
            'gyoza',
            'hamburger',
            'hot and sour soup',
            'hot dog',
            'huevos rancheros',
            'hummus',
            'ice cream',
            'lasagna',
            'lobster bisque',
            'lobster roll sandwich',
            'macaroni and cheese',
            'macarons',
            'miso soup',
            'mussels',
            'nachos',
            'omelette',
            'onion rings',
            'oysters',
            'pad thai',
            'paella',
            'pancakes',
            'panna cotta',
            'peking duck',
            'pho',
            'pizza',
            'pork chop',
            'poutine',
            'prime rib',
            'pulled pork sandwich',
            'ramen',
            'ravioli',
            'red velvet cake',
            'risotto',
            'samosa',
            'sashimi',
            'scallops',
            'seaweed salad',
            'shrimp and grits',
            'spaghetti bolognese',
            'spaghetti carbonara',
            'spring rolls',
            'steak',
            'strawberry shortcake',
            'sushi',
            'tacos',
            'takoyaki',
            'tiramisu',
            'tuna tartare',
            'waffles',
        ] # only for CLIP
        self.class_to_idx = dict(zip(self.classess, range(len(self.classess))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

        self.templates = [
            'a photo of {}, a type of food.',
        ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        # image = PIL.Image.open(image_file).convert("RGB")
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.target_transform:
            raise NotImplementedError
            # label = self.target_transform(label)

        return image, label
