#!/usr/bin/env python3

"""JSON dataset: support CUB, NABrids, Flower, Dogs and Cars"""

import os
import torch.utils.data
import torchvision as tv
import cv2
import numpy as np
from collections import Counter
import json
from bluestar.data.datasets.base_dataset import DatasetBase, register_dataset



class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        assert split in {
            "train",
            "val",
            "test",
        }, "Split '{}' not supported for {} dataset".format(
            split, cfg['data_name'])

        self.cfg = cfg
        self._split = split
        self.name = cfg['data_name']
        self.data_dir = cfg['data_dir']
        self.data_percentage = 1.0
        self._construct_imdb(cfg)

    def get_anno(self):
        anno_path = os.path.join(self.data_dir, "{}.json".format(self._split))
        if "train" in self._split:
            if self.data_percentage < 1.0:
                anno_path = os.path.join(
                    self.data_dir,
                    "{}_{}.json".format(self._split, self.data_percentage)
                )
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)

        with open(anno_path, 'r') as file:
            data = json.load(file, encoding="utf-8")

        return data

    def get_imagedir(self):
        raise NotImplementedError()

    def _construct_imdb(self, cfg):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            im_path = os.path.join(img_dir, img_name)
            self._imdb.append({"im_path": im_path, "class": cont_id})

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES
        # return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        im = cv2.imread(self._imdb[index]["im_path"])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        label = self._imdb[index]["class"]
        index = f"{self._split}{index}"
        sample = {
            "image": im,
            "label": label,
            "id": index
        }
        return sample

    def __len__(self):
        return len(self._imdb)

@register_dataset("CUB_200_2011")
class CUB200Dataset(JSONDataset):
    """CUB_200 dataset."""

    def __init__(self, cfg, split):
        cfg['data_dir']= cfg['data_dir']+"CUB_200_2011/"
        super(CUB200Dataset, self).__init__(cfg, split)
    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")

@register_dataset("stanford-cars")
class CarsDataset(JSONDataset):
    """stanford-cars dataset."""

    def __init__(self, cfg, split):
        cfg['data_dir']= cfg['data_dir']+"stanford-cars/"
        super(CarsDataset, self).__init__(cfg, split)
    def get_imagedir(self):
        return self.data_dir

@register_dataset("stanford-dogs")
class DogsDataset(JSONDataset):
    """stanford-dogs dataset."""

    def __init__(self, cfg, split):
        cfg['data_dir']= cfg['data_dir']+"stanford-dogs/"
        super(DogsDataset, self).__init__(cfg, split)
    def get_imagedir(self):
        return os.path.join(self.data_dir, "Images")

@register_dataset("flowers102")
class FlowersDataset(JSONDataset):
    """flowers dataset."""

    def __init__(self, cfg, split):
        cfg['data_dir']= cfg['data_dir']+"flowers102/"
        super(FlowersDataset, self).__init__(cfg, split)
    def get_imagedir(self):
        return self.data_dir


@register_dataset("nabirds")
class NabirdsDataset(JSONDataset):
    """Nabirds dataset."""

    def __init__(self, cfg, split):
        cfg['data_dir'] = cfg['data_dir'] + "nabirds/"
        super(NabirdsDataset, self).__init__(cfg, split)
    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")

