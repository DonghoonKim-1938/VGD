from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image
import cv2
from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset

@register_dataset("Flowers102")
class Flowers102(DatasetBase):

    _file_dict = {  # filename,
        "label": ("imagelabels.mat",),
        "setid": ("setid.mat", ),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

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
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        self.classes = ['pink primrose',
 'hard-leaved pocket orchid',
'canterbury bells',
'sweet pea',
'english marigold',
'tiger lily',
'moon orchid',
 'bird of paradise',
 'monkshood',
 'globe thistle',
  'snapdragon',
  "colt's foot",
  'king protea',
  'spear thistle',
  'yellow iris',
  'globe-flower',
  'purple coneflower',
  'peruvian lily',
  'balloon flower',
  'giant white arum lily',
  'fire lily',
  'pincushion flower',
  'fritillary',
  'red ginger',
  'grape hyacinth',
  'corn poppy',
  'prince of wales feathers',
  'stemless gentian',
  'artichoke',
  'sweet william',
  'carnation',
  'garden phlox',
  'love in the mist',
  'mexican aster',
  'alpine sea holly',
  'ruby-lipped cattleya',
  'cape flower',
  'great masterwort',
  'siam tulip',
  'lenten rose',
  'barbeton daisy',
  'daffodil',
  'sword lily',
  'poinsettia',
  'bolero deep blue',
  'wallflower',
  'marigold',
  'buttercup',
  'oxeye daisy',
  'common dandelion',
  'petunia',
  'wild pansy',
  'primula',
  'sunflower',
  'pelargonium',
  'bishop of llandaff',
  'gaura',
  'geranium',
  'orange dahlia',
  'pink-yellow dahlia?',
  'cautleya spicata',
  'japanese anemone',
  'black-eyed susan',
  'silverbush',
  'californian poppy',
  'osteospermum',
  'spring crocus',
  'bearded iris',
  'windflower',
  'tree poppy',
  'gazania',
  'azalea',
  'water lily',
  'rose',
  'thorn apple',
  'morning glory',
  'passion flower',
  'lotus',
  'toad lily',
  'anthurium',
  'frangipani',
  'clematis',
  'hibiscus',
  'columbine',
  'desert-rose',
  'tree mallow',
  'magnolia',
  'cyclamen ',
  'watercress',
  'canna lily',
  'hippeastrum ',
  'bee balm',
  'ball moss',
  'foxglove',
  'bougainvillea',
  'camellia',
  'mallow',
  'mexican petunia',
  'bromelia',
  'blanket flower',
  'trumpet creeper',
  'blackberry lily'] #
        # self.classes = [cls.replace(' ', '_') for cls in classes]

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

        self.templates = [
            'a photo of a {}, a type of flower.',
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
