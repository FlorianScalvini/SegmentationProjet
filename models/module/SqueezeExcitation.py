import torch.nn as nn
from functools import partial


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_channels, activation=partial(nn.ReLU, True), scale_activation=nn.Sigmoid):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1, stride=1, bias=True)
        self.act = activation()
        self.fc2 = nn.Conv2d(in_channels=squeeze_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=True)
        self.scale_act = scale_activation()


    def forward(self,x):
        y = self.avgpool(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.scale_act(y)
        return y * x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)