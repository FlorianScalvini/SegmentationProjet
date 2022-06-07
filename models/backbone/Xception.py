import torch
import torch.nn as nn
from models.module.convBnRelu import ConvBN, ConvBNRelu


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.conv = ConvBNRelu(3, 32, 3, 2, 0, bias=False)
        self.conv1 = ConvBNRelu(32, 64, 3, bias=False)
        self.block = Block(in_channels=64, out_channels=128, reps=2, start_relu=False, skip_conv=True)
        self.block2 = Block(in_channels=128, out_channels=256, reps=2, start_relu=True, skip_conv=True)
        self.block3 = Block(in_channels=256, out_channels=728, reps=2, start_relu=True, skip_conv=True)
        self.block4 = Block(in_channels=728, out_channels=728, reps=3,  start_relu=True,  skip_conv=False)
        self.block5 = Block(in_channels=728, out_channels=728, reps=3,  start_relu=True,  skip_conv=False)
        self.block6 = Block(in_channels=728, out_channels=728, reps=3,  start_relu=True,  skip_conv=False)
        self.block7 = Block(in_channels=728, out_channels=728, reps=3,  start_relu=True,  skip_conv=False)
        self.block8 = Block(in_channels=728, out_channels=728, reps=3,  start_relu=True,  skip_conv=False)
        self.block9 = Block(in_channels=728, out_channels=728, reps=3, start_relu=True, skip_conv=False)
        self.block10 = Block(in_channels=728, out_channels=728, reps=3, start_relu=True, skip_conv=False)
        self.block11 = Block(in_channels=728, out_channels=728, reps=3, start_relu=True, skip_conv=False)
        self.block12 = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels=728, out_channels=728),
            nn.BatchNorm2d(num_features=728),
            nn.ReLU(),
            SeparableConv2d(in_channels=728, out_channels=1024),
            nn.BatchNorm2d(num_features=1024),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block12_skip = ConvBN(in_channels=728, out_channels=1024, kernel_size=1, stride=2, bias=False)
        self.sepbn = nn.Sequential(
            SeparableConv2d(in_channels=1024, out_channels=1536),
            nn.BatchNorm2d(num_features=1536),
            nn.ReLU()
        )
        self.sepbn1 = nn.Sequential(
            SeparableConv2d(in_channels=1536, out_channels=2048),
            nn.BatchNorm2d(num_features=2048),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x, backbone=True):
        y = self.conv(x)
        y = self.conv1(y)
        y = self.block(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.block6(y)
        y = self.block7(y)
        y = self.block8(y)
        y = self.block9(y)
        y = self.block10(y)
        y = self.block11(y)
        y_16 = self.block12(y)
        y = self.block12_skip(y) + y_16
        y = self.sepbn(y)
        y_32 = self.sepbn1(y)
        y_global = self.avgpool(y_32)
        if backbone:
            return y_16, y_32, y_global
        else:
            y = self.linear(y_global)
            return y



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,stride=1,padding=1,dilation=1,bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        y = self.conv1(x)
        y = self.pointwise(y)
        return y

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps=3, skip_conv=True, start_relu=True):
        super(Block, self).__init__()
        self.start_relu = start_relu
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reps = reps
        self.skip_conv = skip_conv
        self.layers = self._make_block()
        self.skip_layers = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False)


    def _make_block(self):
        layers = []
        if self.start_relu:
            layers.append(nn.ReLU())
        layers.append(SeparableConv2d(in_channels=self.in_channels, out_channels=self.out_channels))
        layers.append(nn.BatchNorm2d(self.out_channels))

        for i in range(self.reps-1):
            layers.append(nn.ReLU())
            layers.append(SeparableConv2d(in_channels=self.out_channels, out_channels=self.out_channels))
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

if "__main__" == __name__:
    model = Xception()
    input = torch.randn((1,3,224,224))
    pred = model(input)
    print('End')