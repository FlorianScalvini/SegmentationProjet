import torch
import torch.nn as nn
import math
from models.module import *
from models.backbone.Backbone import Backbone


class STDC(Backbone):
    def __init__(self,
                 base=64,
                 layers=None,
                 block_num=4,
                 type="cat",
                 use_conv_last=False):
        super(Backbone, self).__init__()
        if layers is None:
            layers = [4, 5, 3]
        self.feat_channels = [base // 2, base, base * 4, base * 8, base * 16]
        block = None
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.features_channels = [base // 2, base, base * 4, base * 8, base * 16]
        self.conv_last = ConvBNRelu(base*16, max(1024, base*16), 1, 1, padding=0)
        self.x = [nn.Sequential(self.features[:1])]
        self.x.append(nn.Sequential(self.features[1:2]))
        idx = 2
        for i in layers[:-1]:
            self.x.append(nn.Sequential(self.features[idx:idx + i]))
            idx += i
        self.x.append(nn.Sequential(self.features[idx:]))
        self.init_weight()
        return

    def forward(self, x):
        self.feat = [self.x[0](x)]
        for i in range(len(self.x[1:])):
            self.feat.append(self.x[i+1](self.feat[i]))
        if self.use_conv_last:
            self.feat[-1] = self.conv_last(self.feat[-1])
        return tuple(self.feat)


    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvBNRelu(3, base // 2, 3, 2)]
        features += [ConvBNRelu(base // 2, base, 3, 2)]
        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i + 1)), base * int(
                            math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 2)), base * int(
                            math.pow(2, i + 2)), block_num, 1))
        return nn.Sequential(*features)


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvBNRelu(in_planes, out_planes // 2, kernel_size=1,padding=0))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride,padding=1))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride,padding=1))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1)), padding=1))
            else:
                self.conv_list.append(ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx)), padding=1))

    def forward(self, x):
        out_list = []
        out = x
        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvBNRelu(in_planes, out_planes // 2, kernel_size=1, padding=0))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride, padding=1))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride, padding=1))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1)), padding=1))
            else:
                self.conv_list.append(ConvBNRelu(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx)), padding=1))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        out = None
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = torch.cat(out_list, dim=1)
        return out
