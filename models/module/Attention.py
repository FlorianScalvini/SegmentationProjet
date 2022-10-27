# Unified Attention Fusion Module from https://arxiv.org/abs/2204.02681

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module import ConvBNRelu, ConvBN
from math import log

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

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='nearest'):
        super().__init__()

        self.conv_x = ConvBNRelu(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias=False)
        self.conv_out = ConvBNRelu(
            y_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.resize_mode = resize_mode

    def check(self, x, y):
        # print("x dim:",x.ndim)
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        # print("x,y shape:",x.shape, y.shape)
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out

class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='nearest'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNRelu(
                4, 2, kernel_size=3, padding=1, bias=False),
            ConvBN(
                2, 1, kernel_size=3, padding=1, bias=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        # print("x, y shape:",x.shape, y.shape)
        atten = self.avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


    def avg_max_reduce_channel_helper(self, x, use_concat=True):
        # Reduce hw by avg and max, only support single input
        assert not isinstance(x, (list, tuple))
        # print("x before mean and max:", x.shape)
        mean_value = torch.mean(x, dim=1, keepdim=True)
        max_value = torch.max(x, dim=1, keepdim=True)[0]
        # mean_value = mean_value.unsqueeze(0)
        # print("mean max:", mean_value.shape, max_value.shape)

        if use_concat:
            res = torch.cat([mean_value, max_value], dim=1)
        else:
            res = [mean_value, max_value]
        return res


    def avg_max_reduce_channel(self, x):
        # Reduce hw by avg and max
        # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
        if not isinstance(x, (list, tuple)):
            return self.avg_max_reduce_channel_helper(x)
        elif len(x) == 1:
            return self.avg_max_reduce_channel_helper(x[0])
        else:
            res = []
            for xi in x:
                # print(xi.shape)
                res.extend(self.avg_max_reduce_channel_helper(xi, False))
            # print("res:\n",)
            # for it in res:
            #     print(it.shape)
            return torch.cat(res, dim=1)


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


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual




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

