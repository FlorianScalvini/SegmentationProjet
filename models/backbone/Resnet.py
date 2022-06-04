import torch
import torchvision
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

    def forward(self, x):
        return