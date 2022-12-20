import BaseModel
from backbone import *
import torch.nn as nn
from models.module import *
from torch.nn.functional import *


class SegNet(BaseModel):
    def __init__(self, num_classes, freeze_bn=False):
        super(SegNet, self).__init__(num_classes=num_classes)

        self.stage1 = nn.Sequential(
            ConvBNRelu( 3, 64, 3, padding=1),
            ConvBNRelu(64, 64, 3, padding=1))

        self.stage2 = nn.Sequential(
            ConvBNRelu(64, 128, 3, padding=1),
            ConvBNRelu(128, 128, 3, padding=1))

        self.stage3 = nn.Sequential(
            ConvBNRelu(128, 256, 3, padding=1),
            ConvBNRelu(256, 256, 3, padding=1),
            ConvBNRelu(256, 256, 3, padding=1))

        self.stage4 = nn.Sequential(
            ConvBNRelu(256, 512, 3, padding=1),
            ConvBNRelu(512, 512, 3, padding=1),
            ConvBNRelu(512, 512, 3, padding=1))

        self.stage5 = nn.Sequential(
            ConvBNRelu(512, 512, 3, padding=1),
            ConvBNRelu( 512, 512, 3, padding=1),
            ConvBNRelu(512, 512, 3, padding=1))


        self.unstage5 = nn.Sequential(
            ConvBNRelu(512, 512, 3, padding=1),
            ConvBNRelu(512, 512, 3, padding=1),
            ConvBNRelu(512, 512, 3, padding=1))

        self.unstage4 = nn.Sequential(
            ConvBNRelu(512, 512, 3, padding=1),
            ConvBNRelu(512, 512, 3, padding=1),
            ConvBNRelu(512, 256, 3, padding=1))

        self.unstage3 = nn.Sequential(
            ConvBNRelu(256, 256, 3, padding=1),
            ConvBNRelu(256, 256, 3, padding=1),
            ConvBNRelu(256, 128, 3, padding=1))

        self.unstage2 = nn.Sequential(
            ConvBNRelu(128, 128, 3, padding=1),
            ConvBNRelu(128, 128, 3, padding=1),
            ConvBNRelu(128, 64, 3, padding=1))

        self.unstage1 = nn.Sequential(
            ConvBNRelu(64, 64, 3, padding=1),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        if freeze_bn: self.freeze_bn()
        self._init_weight()

    def forward(self, x):
        y = self.stage1(x)
        y, ind1 = self.pool(y)
        size1 = y.shape

        y = self.stage2(y)
        x, ind2 = self.pool(y)
        size2 = y.shape

        y = self.stage3(y)
        y, ind3 = self.pool(y)
        size3 = y.shape

        y = self.stage4(x)
        y, ind4 = self.pool(y)
        size4 = y.shape

        y = self.stage5(x)
        y, ind5 = self.pool(y)
        size5 = y.shape

        y = self.unpool(y, indices=ind5, output_size=size5)
        y = self.deco1(y)

        y = self.unpool(y, indices=ind4, output_size=size4)
        y = self.deco4(y)

        y = self.unpool(y, indices=ind3, output_size=size3)
        y = self.deco3(y)

        y = self.unpool(y, indices=ind2, output_size=size2)
        y = self.deco2(y)

        y = self.unpool(y, indices=ind1, output_size=size1)
        y = self.deco1(y)
        return y

