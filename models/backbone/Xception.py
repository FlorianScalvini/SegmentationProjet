import torch.nn as nn
from models.module.convBnRelu import ConvBN
class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.block =

    def forward(self, x):



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        y = self.conv1(x)
        y = self.pointwise(y)
        return y

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps=3, skip=True, skip_conv=True, start_relu=True):
        super(Block, self).__init__()
        self.skip = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
        self.relu = nn.ReLU()
        self.start_relu = start_relu
        self.skip_conv = skip_conv
        self.skip = skip
        self.skip_layers = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False)
        self.reps = reps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = self._make_block()

    def _make_block(self):
        layers = []
        if self.start_relu:
            layers.append(self.relu)
        layers.append(SeparableConv2d(in_channels=self.in_channels, out_channels=self.out_channels))
        layers.append(nn.BatchNorm2d(self.out_channels))
        for i in range(self.reps-1):
            layers.append(self.relu)
            layers.append(SeparableConv2d(in_channels=self.in_channels, out_channels=self.out_channels))
            layers.append(nn.BatchNorm2d(self.out_channels))
        if self.skip_conv:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        idn = x
        y = self.layers(idn)
        if self.skip:
            if self.skip_conv:
                idn = self.skip_layers(idn)
            y += idn
        return y

