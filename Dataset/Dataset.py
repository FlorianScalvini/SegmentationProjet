import random
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import hflip, rotate, InterpolationMode
import numpy as np
import torch
from transform import Transform

class LabelInfo:
    def __init__(self, name, id, trainId, category, categoryId, hasInstances, ignoreInEval, color):
        self.name = name
        self.id = id
        self.trainId = trainId
        self.category = category
        self.categoryId = categoryId
        self.hasInstances = hasInstances
        self.ignoreInEval = ignoreInEval
        self.color = color


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


class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break
