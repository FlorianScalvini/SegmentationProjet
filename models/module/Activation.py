import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

'''
Script provides functional interface for Mish activation function.
Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
https://arxiv.org/abs/1908.08681v1
'''
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))


class BetaMish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        beta=1.5
        return x * torch.tanh(torch.log(torch.pow((1+torch.exp(x)),beta)))



'''
Swish - https://arxiv.org/pdf/1710.05941v1.pdf
'''
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.



class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, activation=partial(nn.ReLU, True), scale_activation=nn.Sigmoid):
        super(SEModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1, stride=1,
                             bias=True)
        self.act = activation()
        self.fc2 = nn.Conv2d(in_channels=squeeze_channels, out_channels=in_channels, kernel_size=1, stride=1,
                             bias=True)
        self.scale_act = scale_activation()

    def forward(self, x):
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


NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'PReLU': nn.PReLU(),
    'ReLu6': nn.ReLU6(inplace=True),
    'Mish': Mish(),
    'BetaMish': BetaMish(),
    'Swish': Swish(),
    'Hswish': Hswish(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid()
}