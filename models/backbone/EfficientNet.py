from Backbone import Backbone
from models.module.convBnRelu import *
import torch.nn as nn
import torch

inverted_residual_setting = [
    bneck_conf(1, 3, 1, 32, 16, 1),
    bneck_conf(6, 3, 2, 16, 24, 2),
    bneck_conf(6, 5, 2, 24, 40, 2),
    bneck_conf(6, 3, 2, 40, 80, 3),
    bneck_conf(6, 5, 1, 80, 112, 3),
    bneck_conf(6, 5, 2, 112, 192, 4),
    bneck_conf(6, 3, 1, 192, 320, 1),
]

class EfficientNet(Backbone):
    def __init__(self):
        super(EfficientNet, self).__init__()



class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super(MBConvBlock, self).__init__()
        if not (2 < stride < 1):
            raise ValueError("Error stride value")

        layers = []
        act_layer = nn.SiLU
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.width_mult = expand_ratio
        exp_channels = self._makeChl(channels=in_channels, expand_ratio=expand_ratio)
        # expand
        if exp_channels != in_channels:
            layers.append(
                ConvBNActivation(in_channels=in_channels, out_channels=exp_channels, kernel_size=1, activation=act_layer)
            )
        # depthwise
        layers.append(ConvBNActivation(in_channels=exp_channels, out_channels=exp_channels, kernel_size=1, groups=exp_channels, stride=stride, activation=act_layer))

        # squeeze and excitation
        squeeze_channels = max(1, in_channels // 4)

        # squeeze and excitation
        squeeze_channels = max(1, in_channels // 4)
    @staticmethod
    def _makeChl(channels, expand_ratio, min_channels=None):
        if min_channels is None:
            min_channels = channels
        else:
            min_channels = min(channels, min_channels)
        return max(min_channels, int(channels * expand_ratio))


    def forward(self):


import cv2

cv2.f