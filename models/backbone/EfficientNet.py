from models.backbone.Backbone import Backbone
from models.module import ConvBNActivation, SqueezeExcitation, ConvBN, StochasticDepth
import torch.nn as nn
from utils.utils import make_divisible
import torch
from functools import partial
import math



def makeChl(channels, expand_ratio, min_channels=None):
    if min_channels is None:
        min_channels = channels
    else:
        min_channels = min(channels, min_channels)
    return max(min_channels, int(channels * expand_ratio))



class EfficientNet(Backbone):
    def __init__(self, type="b0", stoch_depth_prob=0.2):
        super(EfficientNet, self).__init__()
        type = type.lower()
        if type not in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "v2_s", "v2_m", "v2_l"]:
            raise ValueError("Not implemented model")
        if type[0] == "b":
            config = self._confEfficientNetB(type)
        else:
            config = self._confEfficientNetV2(type)

        total_block_len = 0
        stage_block_id = 0
        for _, _, _, _, _, _, k  in config:
            total_block_len += k
        in_channels = None
        self.layers = []
        for t, ci, co, s, k, exp, n in config:
            if not self.layers:
                self.layers.append(ConvBNActivation(3, ci, kernel_size=3, stride=2, activation=partial(nn.SiLU, True), padding=1))
                in_channels = ci
            block = []
            for _ in range(n):
                sd_prob = stoch_depth_prob * float(stage_block_id) / total_block_len
                if t == "FusedMBConv":
                    block.append(FusedMBConv(in_channels=in_channels, out_channels=co, stride=s, kernel_size=k,
                                              expand_ratio=exp, stoch_depth_prob=sd_prob))
                elif t == "MBConvBlock":
                    block.append(MBConvBlock(in_channels=in_channels, out_channels=co, stride=s, kernel_size=k,
                                              expand_ratio=exp, stoch_depth_prob=sd_prob))
                else:
                    raise ValueError('Unimplemented residual block')
                in_channels = co
                stage_block_id += 1
            self.layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*self.layers)
        channels_classifier = 1280 if type[0] != 'b' else 4 * in_channels

        self.Classifier = nn.Sequential(
            ConvBNActivation(in_channels, channels_classifier, kernel_size=1, activation=partial(nn.SiLU, True), bias=False),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(1),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=channels_classifier, out_features=self.num_classes, bias=True)
        )

        self._init_weight()

    def forward(self, x):
        y = self.layers(x)
        if self._backbone:
            return y
        else:
            return self.Classifier(y)

    @staticmethod
    def _confEfficientNetB(type="b0"):
        config_width_depth = {
            "b0": [1.0, 1.0],
            "b1": [1.0, 1.1],
            "b2": [1.1, 1.2],
            "b3": [1.2, 1.4],
            "b4": [1.4, 1.8],
            "b5": [1.6, 2.2],
            "b6": [1.8, 2.6],
            "b7": [2.0, 3.1],
        }
        config_inv_residual = [
            ["MBConvBlock", 32, 16, 1, 3, 1, 1],
            ["MBConvBlock", 16, 24, 2, 3, 6, 2],
            ["MBConvBlock", 24, 40, 2, 5, 6, 2],
            ["MBConvBlock", 40, 80, 2, 3, 6, 3],
            ["MBConvBlock", 80, 112, 1, 5, 6, 3],
            ["MBConvBlock", 112, 192, 2, 5, 6, 4],
            ["MBConvBlock", 192, 320, 1, 3, 6, 1]
        ]
        width, depth = config_width_depth[type]
        for idx in range(len(config_inv_residual)):
            config_inv_residual[idx][1] = make_divisible(config_inv_residual[idx][1] * width, 8)
            config_inv_residual[idx][2] = make_divisible(config_inv_residual[idx][2] * width, 8)
            config_inv_residual[idx][6] = int(math.ceil(config_inv_residual[idx][6] * depth))
        return config_inv_residual



    @staticmethod
    def _confEfficientNetV2(type="v2_m"):
        config_inv_residual = None
        if type == "v2_s":
            config_inv_residual = [
                ["FusedMBConv", 24, 24, 1, 3, 1, 2],
                ["FusedMBConv", 24, 48, 2, 3, 4, 4],
                ["FusedMBConv", 48, 64, 2, 3, 4, 4],
                ["MBConvBlock", 64, 128, 2, 3, 4, 6],
                ["MBConvBlock", 128, 160, 1, 3, 6, 9],
                ["MBConvBlock", 160, 256, 2, 3, 6, 15],
            ]
        elif type == "v2_m":
            config_inv_residual = [
                ["FusedMBConv", 24, 24, 1, 3, 1, 3],
                ["FusedMBConv", 24, 48, 2, 3, 4, 5],
                ["FusedMBConv", 48, 80, 2, 3, 4, 5],
                ["MBConvBlock", 80, 160, 2, 3, 4, 7],
                ["MBConvBlock", 160, 176, 1, 3, 6, 14],
                ["MBConvBlock", 176, 304, 2, 3, 6, 28],
                ["MBConvBlock", 304, 512, 1, 3, 6, 5],
            ]

        elif type == "v2_l":
            config_inv_residual = [
                ["FusedMBConv", 32, 32, 1, 3, 1, 4],
                ["FusedMBConv", 32, 64, 2, 3, 4, 7],
                ["FusedMBConv", 64, 96, 2, 3, 4, 7],
                ["MBConvBlock", 96, 192, 2, 3, 4, 10],
                ["MBConvBlock", 192, 224, 1, 3, 6, 19],
                ["MBConvBlock", 224, 384, 2, 3, 6, 25],
                ["MBConvBlock", 384, 640, 1, 3, 6, 7],
            ]
        else:
            raise ValueError("Not implemented network EfficientNet2" + str(type))
        return config_inv_residual



class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio, stride, stoch_depth_prob=0.2):
        super(FusedMBConv, self).__init__()
        if not (1 <= stride <= 2):
            raise ValueError("Error stride value")

        layers = []
        act_layer = partial(nn.SiLU, True)
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
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, stoch_depth_prob=0.2):
        super(MBConvBlock, self).__init__()
        if not (1 <= stride <= 2):
            raise ValueError("Error stride value")

        layers = []
        act_layer = partial(nn.SiLU, True)
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
        layers.append(ConvBNActivation(in_channels=exp_channels, out_channels=exp_channels, kernel_size=kernel_size,
                                       groups=exp_channels, stride=stride, padding=kernel_size // 2,
                                       activation=act_layer, bias=False))

        # squeeze and excitation
        squeeze_channels = max(1, in_channels // 4)
        layers.append(SqueezeExcitation(in_channels=exp_channels,
                                        squeeze_channels=squeeze_channels, activation=partial(nn.SiLU, True)))

        # project
        layers.append(ConvBN(in_channels=exp_channels, out_channels=out_channels, kernel_size=1, bias=False))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stoch_depth_prob, "row")

    def forward(self, x):
        y = self.block(x)
        if self.use_residual:
            y = self.stochastic_depth(y)
            y += x
        return y


if __name__ == "__main__":
    import torchsummary
    import torchvision.models
    mdl = EfficientNet(type="b0")
    mdl = mdl.cuda()
    torchsummary.summary(mdl, (3, 224, 224))