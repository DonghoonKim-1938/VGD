import torch.utils.data
import os
from bluestar.data.datasets import LmdbReader
DATASETS = [
    'caltech101',
    'cifar100_tf',
    'dtd',
    'oxford_flowers102',
    'oxford_iiit_pet',
    'patch_camelyon',
    'sun397',
    'svhn_cropped',
    'resisc45',
    'eurosat',
    'dmlab',
    'kitti',
    'smallnorb',
    'dsprites',
    'clevr',
    'diabetic_retinopathy_detection',
    'CUB_200_2011',
    'flowers102',
    'nabirds',
    'stanford-cars',
    'stanford-dogs',
]
class lmdbDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split, transform= None):
        assert split in {
            "train",
            "val",
            "test",
            "trainval"
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg['data_name'])
        if cfg['data_name'] not in DATASETS:
            raise ValueError("Unknown type for \"dataset\" field: {}".format(
                type(cfg['data_name'] )))
        self.split = split
        lmdb_dir = os.path.join(cfg['data_dir'], f"{cfg['data_name']}_{split}.lmdb")
        self.instances = LmdbReader(lmdb_dir)
        self.transform = transform
    def __getitem__(self, idx):
        image, label = self.instances[f"{self.split}{idx}"].values()
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.instances)
