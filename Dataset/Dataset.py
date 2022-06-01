import random
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import hflip, rotate, InterpolationMode
import numpy as np
import torch
from transform import Transform

class BaseDataSet(Dataset):
    def __init__(self, root, num_classes, transforms=Transform()):
        self.num_classes = num_classes
        self.root = root
        self.transforms = Transform(transforms=transforms)
        self.files = []
        self.palette = None
        self.mapping = None
        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

    def _set_files(self):
        raise NotImplementedError

    def _load_data(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label = self._load_data(index)
        image, label = self.transforms(image, label)
        if self.mapping is not None:
            label = self.encode_labels(label)
        return image, label.squeeze().long()

    def encode_labels(self,mask):
        label_mask = torch.zeros_like(mask)
        for k in self.mapping:
            label_mask[mask == k] = self.mapping[k]
        return label_mask

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str