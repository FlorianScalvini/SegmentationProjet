import torch
import torch.nn as nn
import torch.nn.functional
from BaseModel import BaseModel
from models.module import *
from models.bisenetv2 import StemBlock
from models.backbone.EfficientNet import MBConvBlock
class NewModel(nn.Module):
    def __init__(self, num_classes):
        self.db = DetailBranch()
        self.sb = SemanticBranch()
        self.
        self.aux_head1 = SegHead(C1, C1, num_classes)
        self.aux_head2 = SegHead(C3, C3, num_classes)
        self.aux_head3 = SegHead(C4, C4, num_classes)
        self.aux_head4 = SegHead(C5, C5, num_classes)
        self.head = SegHead(mid_channels, mid_channels, num_classes)
        self.init_weight()
        return

    def forward(self,  x_color, x_depth):
        dfm = self.db( x_color, x_depth)
        feat1, feat2, feat3, feat4, sfm = self.sb(x_color)
        out = self.head(self.bga(dfm, sfm))
        if not self.training:
            out_list = out
        else:
            out_1 = self.aux_head1(feat1)
            out_2 = self.aux_head2(feat2)
            out_3 = self.aux_head3(feat3)
            out_4 = self.aux_head4(feat4)
            out_list = [out, out_1, out_2, out_3, out_4]
        out_list = [nn.functional.interpolate(out_list, x_color.shape[2:], mode='bilinear', align_corners=self.align_corners) for out in out_list]
        return out_list



class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self
        return

    def forward(self, x_color, x_depth):
        return

class SemanticBranch(nn.Module):
    def __init__(self, channel_stage):
        super(SemanticBranch, self).__init__()
        C1, C2, C3, C4 = channel_stage
        self.stem = StemBlock(out_channels=C1)
        self.block_1 = nn.Sequential(
            MBConvBlock(in_channels=C1, out_channels=C2, expand_ratio= , stride= , kernel_size=, stoch_depth_prob=0.2),
            MBConvBlock(in_channels=C1, out_channels=C2, expand_ratio= , stride= , kernel_size=, stoch_depth_prob=0.2)
        )
        self.block_2 = None
        self.block_3 = None
        self.block_4 = None
        return

    def forward(self, x):
        y = self.stem(x)
        y =
        return

class SegHead(nn.Module):
    def __init__(self, in_dim, mid_dim, num_classes):
        super(SegHead, self).__init__()
        self.conv1 = ConvBNRelu(in_channels=in_dim, out_channels=mid_dim, kernel_size=3, padding=0, bias=True, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(in_channels=mid_dim, out_channels=num_classes, kernel_size=1, stride=1, padding=0,
                               bias=False)
    def forward(self, x):
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return y


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNRelu(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.nn.functional.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
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