import os
import pickle
from typing import Any, Callable, Optional, Tuple
import cv2

from bluestar.utils.dataset_utils import mkdir_if_missing
from bluestar.data.datasets.base_domian_adaptation import DomainAdaptationDatasetBase

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


class Caltech101(DomainAdaptationDatasetBase):

    dataset_dir = "caltech-101"

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            cfg = None,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.cfg = cfg

        root = os.path.abspath(os.path.expanduser(self.root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg["num_shots"]
        if num_shots >= 1:
            seed = cfg["seed"]
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # subsample = cfg.DATASET.SUBSAMPLE_CLASSES # TODO 아마 무조건 all이어서 신경 안써도 될듯??
        # train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        #super().__init__(train_x=train, val=val, test=test)
        self.num_classes = self.get_num_classes(train)
        self.lab2cname, self.classnames = self.get_lab2cname(train)

        if self.split == "train":
            self.data_source = train
        elif self.split =="val":
            self.data_source = val
        else:
            self.data_source = test

    def __len__(self) -> int:
        return len(self.data_source)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self.data_source[idx].impath, self.data_source[idx].label
        # image = PIL.Image.open(image_file).convert("RGB")
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
