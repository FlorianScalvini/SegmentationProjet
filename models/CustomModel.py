import torch
import torchvision
from models.module import *
from models.BaseModel import BaseModel


class CustomModel(BaseModel):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__(num_classes=num_classes)


    def forward(self, x):


