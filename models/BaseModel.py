import logging
import torch.nn as nn
import numpy as np
import models.backbone as backbone
import torchvision
from utils.torchsummary import summary
import models.backbone

class BaseModel(nn.Module):
    def __init__(self, num_classes, pretrained=None, backbone=None):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.num_classes = num_classes
        self.pretrained = pretrained
        if backbone is not None:
            self.backbone = backbone


    def forward(self, x):
        raise NotImplementedError

    def __str__(self):
        nbr_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

    def unfreeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.train()