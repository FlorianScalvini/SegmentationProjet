import torch
import torch.nn as nn
from models.module.conv import ConvBNRelu, ConvBN

"""
    Bisenetv1 Module : https://arxiv.org/pdf/1808.00897.pdf
    ARM : In context path, a ARM module is used to refune the feature of each stage
"""

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.convbn = ConvBN(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.sigmoid_ARM = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        arm = nn.functional.avg_pool2d(feat, feat.size()[2:])
        arm = self.convbn(arm)
        arm = self.sigmoid_ARM(arm)
        out = torch.mul(feat, arm)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params