import torch.nn as nn
from convBnRelu import *

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super(InvertedResidual, self).__init__()
        self.conv = ConvBNActivation(in_channels=in_channels, out_channels=out_channels, activation=nn.ReLU6(), kernel_size=1, stride=1, bias=False)
        self.dwise = ConvBNActivation(in_channels=in_channels, out_channels=in_channels*expansion, activation=nn.ReLU6(), kernel_size=1, stride=stride, bias=False)


        return

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        if self.act is not None:
            y = self.act(y)
        return y

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
