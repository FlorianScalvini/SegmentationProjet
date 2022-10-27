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
    def __init__(self, transforms, root, split='train', *args):
        super(Mapillary, self).__init__(root=root, num_classes=3, transforms=transforms, ignore_label=255)
        if split not in ['training', 'validation', 'test']:
            raise ValueError(
                "mode should be 'training', 'validation' or 'validation', but got {}.".format(
                    split))
        self.root = os.path.join(root, split)
        self.file_list = list()
        self.split = split.lower()
        img_files = sorted(glob.glob(os.path.join(self.root, "images") + '\\*.jpg', recursive=True))
        label_files = sorted(glob.glob(os.path.join(self.root, "v2.0\\labels_convert") + '\\*.png', recursive=True))

        self.files = [[img_path, label_path] for img_path, label_path in zip(img_files, label_files)]

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        return image, label



