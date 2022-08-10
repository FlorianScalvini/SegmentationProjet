# Unified Attention Fusion Module from https://arxiv.org/abs/2204.02681

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module import *
from models.module import ConvBNRelu, ConvBN

class UAFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, in_channels_Low, in_channels_High, out_channels, kernel_size=3, am_type="channel",resize_mode='bilinear', avgMean=True):
        super(UAFM, self).__init__()
        self.resize_mode = resize_mode
        if am_type == "channel":
            self.attenModule = ChannelAM(in_channels_High, avgMean)
        elif am_type == "spatial":
            self.attenModule = SpatialAM(avgMean)
        else:
            raise ValueError("Not implemented Attention Module type")
        self.conv_low = ConvBNRelu(in_channels_Low, in_channels_High, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv_out = ConvBNRelu(in_channels_High, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.resize_mode = resize_mode

    def forward(self, fLow, fHigh):
        fLow = self.conv_low(fLow)
        fUp = nn.functional.interpolate(fHigh, fLow[:2].shape, mode=self.resize_mode)
        alpha = self.attenModule(fLow,fUp)
        out = fUp * alpha + fLow *(1-alpha)
        return out


class ChannelAM(nn.Module):
    def __init__(self, channels, avgmax=True):
        super(ChannelAM, self).__init__()
        if avgmax:
            self.atten = meanMaxReduceHW()
            self.conv = nn.Sequential(
                ConvBNRelu(4 * channels, channels // 2, kernel_size=3, padding=1, bias=False),
                ConvBN(2 * channels, 1 * channels, kernel_size=3, padding=1, bias=False))
        else:
            self.atten = meanReduceHW()
            self.conv = nn.Sequential(
                ConvBNRelu(2 * channels, channels // 2, kernel_size=3, padding=1, bias=False),
                ConvBN(channels // 2, channels, kernel_size=3, padding=1, bias=False))

    def forward(self, x, y):
        out = self.atten(x, y)
        out = self.conv(out)
        out = nn.functional.sigmoid(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class avgPoolReduceChannel(nn.Module):
    def __init__(self):
        super(avgPoolReduceChannel, self).__init__()
    def forward(self, x, y):
        mean_x = torch.mean(input=x, keepdim=True, dim=[2,3])
        mean_y = torch.mean(input=y, keepdim=True, dim=[2,3])
        out = torch.cat([mean_x, mean_y])
        return out

class avgMaxPoolReduceChannel(nn.Module):
    def __init__(self):
        super(avgMaxPoolReduceChannel, self).__init__()

    def forward(self, x, y):
        mean_x = torch.mean(input=x, keepdim=True, dim=[2,3])
        max_x = torch.amax(input=x, keepdim=True, dim=[2,3])
        mean_y = torch.mean(input=y, keepdim=True, dim=[2,3])
        max_y = torch.amax(input=y, keepdim=True, dim=[2,3])
        out = torch.cat([mean_x, max_x, mean_y, max_y])
        return out

class meanReduceHW(nn.Module):
    def __init__(self):
        super(meanReduceHW, self).__init__()

    def forward(self, x, y):
        out_x = torch.mean(input=x, keepdim=True, dim=1)
        out_y = torch.mean(input=y, keepdim=True, dim=1)
        out = torch.cat([out_x, out_y])
        return out

class meanMaxReduceHW(nn.Module):
    def __init__(self):
        super(meanMaxReduceHW, self).__init__()

    def forward(self, x, y):
        mean_x = torch.mean(input=x, keepdim=True, dim=1)
        max_x = torch.amax(input=x, keepdim=True, dim=1)
        mean_y = torch.mean(input=y, keepdim=True, dim=1)
        max_y = torch.amax(input=y, keepdim=True, dim=1)
        out = torch.cat([mean_x, max_x, mean_y, max_y])
        return out
