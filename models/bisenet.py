import torch.nn.functional
from BaseModel import BaseModel
from models.module import *


class Bisenet(BaseModel):
    def __init__(self, num_classes, pretrained=None, lambd=0.25, align_corners=True, *args, **kwargs):
        super(Bisenet, self).__init__(num_classes=num_classes, pretrained=pretrained, backbone=None)

class ContextPath(nn.Module):
    def __init__(self, backbone):
        super(ContextPath, self).__init__()
        self.backbone = backbone
        self.b

class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv = ConvBNRelu(in_channels=3, out_channels=64, stride=2, kernel_size=1, padding=1)
        self.conv2 = ConvBNRelu(in_channels=64, out_channels=128, stride=2, kernel_size=1, padding=1)
        self.conv3 = ConvBNRelu(in_channels=128, out_channels=256, stride=2, kernel_size=1, padding=1)

    def forward(self, x):
        y = self.conv(x)
        y = self.conv2(y)
        y = self.conv3(y)
        return  y