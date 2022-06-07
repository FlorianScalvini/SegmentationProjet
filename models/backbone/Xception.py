import torch.nn as nn
from models.module.convBnRelu import ConvBN
class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.conv = ConvBN(3, 32, 3, 2, 0, bias=False)
        nn.ReLU(inplace=True),
        self.block = Block(in_channels=64, out_channels=128, reps=2, start_relu=False, skip_conv=True),
        self.block2 = Block(in_channels=128, out_channels=256, reps=2, start_relu=True, skip_conv=True),
        self.block3 = Block(in_channels=256, out_channels=728, reps=2, start_relu=True, skip_conv=True)
        self.block4 = Block(in_channels=728, out_channels=728, reps=3,  start_relu=True,  skip_conv=False)
        self.block5 = Block(in_channels=728, out_channels=728, reps=3,  start_relu=True,  skip_conv=False)
        self.block6 = Block(in_channels=728, out_channels=728, reps=3,  start_relu=True,  skip_conv=False)
        self.block7 = Block(in_channels=728, out_channels=728, reps=3,  start_relu=True,  skip_conv=False)
        self.block8 = Block(in_channels=728, out_channels=728, reps=3,  start_relu=True,  skip_conv=False)
        self.block9 = Block(in_channels=728, out_channels=728, reps=3, start_relu=True, skip_conv=False)
        self.block10 = Block(in_channels=728, out_channels=728, reps=3, start_relu=True, skip_conv=False)
        self.block11 = Block(in_channels=728, out_channels=728, reps=3, start_relu=True, skip_conv=False)
        self.block12 =

        self.blockExit = Block(in_channels=728, out_channels=1024, start_relu=True, skip_conv=True)
        self.

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
    def __init__(self, in_channels, out_channels, reps=3, skip_conv=True, start_relu=True, late=False):
        super(Block, self).__init__()
        self.skip = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
        self.relu = nn.ReLU()
        self.start_relu = start_relu
        self.skip_conv = skip_conv
        self.skip_layers = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False)
        self.reps = reps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.late = late
        self.layers = self._make_block()

    def _make_block(self):
        layers = []
        if self.start_relu:
            layers.append(self.relu)
        if not self.late:
            layers.append(SeparableConv2d(in_channels=self.in_channels, out_channels=self.out_channels))
            layers.append(nn.BatchNorm2d(self.out_channels))
            chl = self.out_channels
        else:
            layers.append(SeparableConv2d(in_channels=self.in_channels, out_channels=self.in_channels))
            layers.append(nn.BatchNorm2d(self.out_channels))
            chl = self.in_channels
        for i in range(self.reps-1):
            layers.append(self.relu)
            layers.append(SeparableConv2d(in_channels=chl, out_channels=self.out_channels))
            layers.append(nn.BatchNorm2d(self.out_channels))
        if self.skip_conv:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        idn = x
        y = self.layers(idn)
        if self.skip_conv:
            idn = self.skip_layers(idn)
        y += idn
        return y

