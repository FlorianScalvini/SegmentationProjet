from models.backbone.Backbone import Backbone
from models.module import ConvBNActivation, SqueezeExcitation, ConvBN, StochasticDepth
import torch.nn as nn
import torch
from functools import partial
import math



def makeChl(channels, expand_ratio, min_channels=None):
    if min_channels is None:
        min_channels = channels
    else:
        min_channels = min(channels, min_channels)
    return max(min_channels, int(channels * expand_ratio))



class HarDNet(Backbone):
    def __init__(self, type="b0", logits_out=None):
        super(HarDNet, self).__init__()
        if logits_out is None:
            self.logits_out = None
        else:
             self.logits_out = logits_out

    def forward(self, x):
        return



if __name__ == "__main__":
    import torchsummary
    import torchvision.models
    mdl = HarDNet(type="b1").cuda()
    mdl.classifier()
    torchsummary.summary(mdl, (3, 224, 224))