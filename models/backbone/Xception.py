import torch.nn as nn

class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):



class SeperableConv(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,stride=1,padding=0,dilation=1,bias=False):
        super(SeperableConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        y = self.conv1(x)
        y = self.pointwise(y)
        return y

class MiddleFlow(nn.Module):
    def __init__(self, channels, number=8):
        super(MiddleFlow, self).__init__()
        self.channels = channels
        self.number = number
        self.block = self._makeblock()

    def _makeblock(self):
        layers = []
        for i in range(3):
            layers.append(nn.Sequential(
                nn.ReLU(),
                SeperableConv(in_channels=self.channels, out_channels=self.channels),
                nn.BatchNorm2d(num_features=self.channels)
            ))

    def forward(self, x):
        for
        return y

class Block2RS_M(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block2RS_M, self).__init__()
        self.sep1 = SeperableConv(in_channels=in_channels, out_channels=out_channels)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.sep2 = SeperableConv(in_channels=out_channels, out_channels=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.conv = nn.C

    def forward(self, x):