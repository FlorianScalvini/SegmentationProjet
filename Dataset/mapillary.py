# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob

import random
from torch.utils.data import Dataset
from Dataset.Dataset import BaseDataSet, Label
import PIL.Image as Image




class Mapillary(BaseDataSet):
    def __init__(self, transforms, root, config_json, split='train', depth=False, *args):
        super(Mapillary, self).__init__(root=root, num_classes=19, transforms=transforms, labels=labels, depth=depth,
                                         ignore_label=255)
        self.file_list = list()
        self.split = split.lower()
        self.ignore_index = 0
        img_dir = os.path.join(self.root, 'leftImg8bit')
        label_dir = os.path.join(self.root, 'gtFine')
        if self.root is None or not os.path.isdir(self.root) or not os.path.isdir(img_dir) \
                or not os.path.isdir(label_dir):
            raise ValueError("The dataset is not Found.")
        self._set_files()

    def _set_files(self):
        assert (self.split in ['train', 'val'])
        label_path = os.path.join(self.root, 'gtFine', self.split)
        image_path = os.path.join(self.root, 'leftImg8bit', self.split)
        assert os.listdir(image_path) == os.listdir(label_path)
        image_paths = glob.glob(image_path + '/**/*.png', recursive=True)
        label_paths = glob.glob(label_path + '/**/*gtFine_labelIds.png', recursive=True)
        image_paths = sorted(image_paths)
        label_paths = sorted(label_paths)
        if self.depth:
            depth_path = os.path.join(self.root, 'disparity', self.split)
            depth_paths = glob.glob(depth_path + '/**/*.png', recursive=True)
            depth_paths = sorted(depth_paths)
            self.files = list(zip(image_paths, depth_paths, label_paths))
        else:
            self.files = list(zip(image_paths, label_paths))


    def _load_data(self, index):
        if self.depth:
            image_path, depth_label, label_path = self.files[index]
            depth = Image.open(depth_label).convert("L")
        else:
            image_path, label_path = self.files[index]
            depth = None
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        return image, depth, label



