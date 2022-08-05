import torch
import torch.nn as nn
import torch.nn.functional
from BaseModel import BaseModel
from models.module import *

class NewModel(BaseModel):
    def __init__(self, backbone):
        self.backbone = backbone
        return

    def forward(self, x, d):
        self.backbone
        return




class BGALayer(nn.Module):
    """
        Bilateral Guided Aggregation Layer : Fuse information from Sementic Branch & Detail branch
    """
    def __init__(self, out_channels, align_corners=True):
        super(BGALayer, self).__init__()
        self.align_corners = align_corners
        self.lDepthWiseConv = nn.Sequential(
            ConvBN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.rDepthWiseConv = nn.Sequential(
            ConvBN(in_channels=out_channels , out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.rConv = ConvBN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.convAvg = nn.Sequential(
            ConvBN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )

        self.convOut = ConvBN(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.lDepthWiseConv(x_d)
        left2 = self.convAvg(x_d)
        right2 = self.rDepthWiseConv(x_s)
        right1 = self.rConv(x_s)
        right1 = nn.functional.interpolate(right1, size=dsize, mode='bilinear', align_corners=self.align_corners)
        left = left1 * self.sigmoid(right1)
        right = left2 * right2
        right = nn.functional.interpolate(right, size=dsize, mode='bilinear', align_corners=self.align_corners)
        out = self.conv(left + right)
        return out

