import torch
import torchvision
from models.module import *
from models.BaseModel import BaseModel


class CSPVovnet(nn.Module):
    def __init__(self):
        super(CSPVovnet, self).__init__()


    def forward(self, x):
        return

class MS_CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(MS_CAM, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNRelu(in_channels=in_channels, out_channels=int(in_channels/reduction_ratio), kernel_size=1),
            ConvBN(in_channels=int(in_channels/reduction_ratio), out_channels=in_channels, kernel_size=1)
        )
        self.branch_2 = nn.Sequential(
            ConvBNRelu(in_channels=in_channels, out_channels=int(in_channels / reduction_ratio), kernel_size=1),
            ConvBN(in_channels=int(in_channels / reduction_ratio), out_channels=in_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.branch_1(x)
        y2 = self.branch_2(x)
        y = y1 + y2
        y = self.sigmoid(y)
        y = y * x
        return y

class AFF(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(AFF, self).__init__()
        self.ms_cam = MS_CAM(in_channels=in_channels, reduction_ratio=reduction_ratio)

    def forward(self, x, x1):
        y = x + x1
        y_l = x * y
        y_r = (1 - y) * x1
        return y_l + y_r


class iAFF(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(iAFF, self).__init__()
        self.aff = AFF(in_channels=in_channels, reduction_ratio=reduction_ratio)
        self.ms_cam = MS_CAM(in_channels=in_channels, reduction_ratio=reduction_ratio)

    def forward(self, x, x1):
        y = self.aff(x, x1)
        y_l = y * x
        y_r = (1-y) * x1
        return y_l + y_r

class PAPPM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
            BatchNorm(in_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        )

        self.scale_process = nn.Sequential(
            BatchNorm(mid_channels * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 4, mid_channels * 4, kernel_size=3, padding=1, groups=4, bias=False),
        )

        self.compression = nn.Sequential(
            BatchNorm(mid_channels * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 5, out_channels, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            BatchNorm(in_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        out = self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)
        return out


class SegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super(SegHead, self).__init__()
        self.module = nn.Sequential(ConvBNRelu(in_channels=in_channels, out_channels=mid_channels, kernel_size=3,
                                               padding=1, stride=1, bias=False),
                                    nn.Conv2d(in_channels=mid_channels, out_channels=num_classes, kernel_size=1,
                                              stride=1, padding=0, bias=False))
    def forward(self, x):
        return self.module(x)


class CustomModel(BaseModel):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__(num_classes=num_classes)
        self.conv_1 = nn.Sequential(ConvBNRelu(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                                    ConvBNRelu(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False))
        self.pappm = PAPPM(in_channels=128, mid_channels=128, out_channels=128)
        self.iaff = iAFF(in_channels=128)
        self.seghead = SegHead(in_channels=128, mid_channels=128, num_classes=num_classes)

    def forward(self, x):
        y = self.conv_1(x)
        y = self.iaff(y_c, y_d)
        y = self.seghead(y)
        return y



