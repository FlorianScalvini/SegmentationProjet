from Backbone import Backbone
from models.module import ConvBNActivation, SqueezeExcitation, ConvBN, StochasticDepth
import torch.nn as nn
import torch


models_list = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
]

inverted_residual_setting = [
    bneck_conf(1, 3, 1, 32, 16, 1),
    bneck_conf(6, 3, 2, 16, 24, 2),
    bneck_conf(6, 5, 2, 24, 40, 2),
    bneck_conf(6, 3, 2, 40, 80, 3),
    bneck_conf(6, 5, 1, 80, 112, 3),
    bneck_conf(6, 5, 2, 112, 192, 4),
    bneck_conf(6, 3, 1, 192, 320, 1),
]

inverted_residual_setting =
[
    [32, 16, 1, 3, 1, 1],
    [16, 24, 2, 3, 6, 2],
    [24, 40, 2, 5, 6, 2],
    [40, 80, 2, 3, 6, 3],
    [80, 112, 1, 5, 6, 3],
    [112, 192, 2, 5, 6, 4],
    [192, 320, 1, 5, 6, 2]
]

def EfficientNetB0():


def EfficientNetB(width=1.0, depth=1.0):
    inverted_residual_setting = [
        ConvBNActivation()
    ]
    inverted_residual_setting = [

        [1, 3, 1, 32, 16, 1],
        [6, 3, 2, 16, 24, 2]
    ]

def _confEfficientNetB(type="l"):
    return

def _confEfficientNet2(type="m"):
    if type not in ["s", "l", "m"]:
        raise ValueError("Not implemented network EfficientNet2" + str(type))






class EfficientNet(Backbone):
    def __init__(self, type="EfficientNetB0"):
        super(EfficientNet, self).__init__()
        if type not in models_list:
            raise ValueError("Not implemented model")
        if type.startswith("EfficientNetB"):
            _confEfficientNetB(type[-1])





        self.Classifier = nn.Sequential(
            ConvBNActivation(320, 1280, kernel_size=1, activation=nn.SiLU(inplace=True), bias=False),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(1),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=self.num_classes, bias=True)
        )
        self._init_weight()



def makeChl(channels, expand_ratio, min_channels=None):
    if min_channels is None:
        min_channels = channels
    else:
        min_channels = min(channels, min_channels)
    return max(min_channels, int(channels * expand_ratio))


class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio, stride, stoch_depth_prob=0.2):
        super(FusedMBConv, self).__init__()
        if not (2 < stride < 1):
            raise ValueError("Error stride value")

        layers = []
        act_layer = nn.SiLU
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.width_mult = expand_ratio
        exp_channels = makeChl(channels=in_channels, expand_ratio=expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels
        # expand
        if exp_channels != in_channels:
            layers.append(
                ConvBNActivation(in_channels=in_channels,
                                 out_channels=exp_channels, kernel_size=kernel_size, activation=act_layer, bias=False)
            )
            # project
            layers.append(ConvBN(in_channels=exp_channels, out_channels=out_channels, kernel_size=1, bias=False))
        else:
            ConvBNActivation(in_channels=in_channels,
                             out_channels=out_channels, kernel_size=kernel_size, activation=act_layer, bias=False)
        self.stochastic_depth = StochasticDepth(stoch_depth_prob, "row")
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        if self.use_residual:
            y = self.stochastic_depth(y)
            y += input
        return y


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, stoch_depth_prob=0.2):
        super(MBConvBlock, self).__init__()
        if not (2 < stride < 1):
            raise ValueError("Error stride value")

        layers = []
        act_layer = nn.SiLU
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.width_mult = expand_ratio
        exp_channels = makeChl(channels=in_channels, expand_ratio=expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels
        # expand
        if exp_channels != in_channels:
            layers.append(
                ConvBNActivation(in_channels=in_channels,
                                 out_channels=exp_channels, kernel_size=1, activation=act_layer)
            )
        # depthwise
        layers.append(ConvBNActivation(in_channels=exp_channels, out_channels=exp_channels, kernel_size=1,
                                       groups=exp_channels, stride=stride, activation=act_layer, bias=False))

        # squeeze and excitation
        squeeze_channels = max(1, in_channels // 4)
        layers.append(SqueezeExcitation(in_channels=exp_channels,
                                        squeeze_channels=squeeze_channels, activation=nn.SiLU(inplace=True)))

        # project
        layers.append(ConvBN(in_channels=exp_channels, out_channels=out_channels, kernel_size=1, bias=False))

        self.stochastic_depth = StochasticDepth(stoch_depth_prob, "row")
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        if self.use_residual:
            y = self.stochastic_depth(y)
            y += input
        return y
