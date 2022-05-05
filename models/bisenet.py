import torch.nn.functional
from BaseModel import BaseModel
from models.module import *


class Bisenet(BaseModel):
    def __init__(self, num_classes, pretrained=None, lambd=0.25, align_corners=True):
        super(Bisenet, self).__init__(num_classes=num_classes, pretrained=pretrained)
