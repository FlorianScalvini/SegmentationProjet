import torch.nn as nn
import torch


class ConvBNActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True,
                 activation=None, groups=1, dilation=1):
        super(ConvBNActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups, dilation=dilation)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = activation
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

class ConvBN(ConvBNActivation):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, dilation=1, bias=True):
        super(ConvBN, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, bias=bias, activation=None,
                                     groups=groups, dilation=dilation)


class ConvBNRelu(ConvBNActivation):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, bias=True, dilation=1):
        super(ConvBNRelu, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride, padding=padding,
                                         bias=bias, activation=nn.ReLU(), groups=groups, dilation=dilation)

