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




class Korean(BaseDataSet):
    def __init__(self, transforms, root, split='train', *args):
        super(Korean, self).__init__(root=root, num_classes=7, transforms=transforms, ignore_label=7)
        if split not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    split))
        self.root = root
        self.file_list = list()
        self.split = split.lower()
        self.files = list()
        file_path = os.path.join(self.root, split + '.txt')
        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(' ')
                if len(items) != 2:
                    image_path = os.path.join(self.root, items[0])
                    grt_path = None
                else:
                    image_path = os.path.join(self.root, items[0])
                    grt_path = os.path.join(self.root, items[1])
                self.files.append([image_path, grt_path])

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        return image, label



