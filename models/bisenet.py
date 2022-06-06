import torch.nn.functional
from BaseModel import BaseModel
from models.module import *
from torchvision.models.resnet import ResNet

class Bisenet(BaseModel):
    def __init__(self, num_classes, backbone=ResNet, pretrained=None, lambd=0.25, align_corners=True, *args, **kwargs):
        if type(backbone=ResNet or )
        super(Bisenet, self).__init__(num_classes=num_classes, pretrained=pretrained, backbone=None)
