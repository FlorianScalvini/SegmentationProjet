import torch
import torch.nn as nn
import torch.nn.functional
from BaseModel import BaseModel
from models.module import *


class BisenetV2(BaseModel):
    def __init__(self, num_classes, pretrained=None, lambd=0.25, align_corners=True, *args, **kwargs):
        super(BisenetV2, self).__init__(num_classes=num_classes, pretrained=pretrained, backbone=None)
        C1, C2, C3 = 64, 64, 128
        db_channels = (64, 64, 128)
        C1, C3, C4, C5 = int(C1 * lambd), int(C3 * lambd), 64, 128
        sb_channels = (C1, C3, C4, C5)
        mid_channels = 128

        self.db = DetailBranch(db_channels)
        self.sb = SegmenticBranch(sb_channels)
        self.bga = BGALayer(mid_channels, align_corners)
        self.aux_head1 = SegHead(C1, 64, num_classes)
        self.aux_head2 = SegHead(C3, 64, num_classes)
        self.aux_head3 = SegHead(C4, 64, num_classes)
        self.aux_head4 = SegHead(C5, 64, num_classes)
        self.head = SegHead(mid_channels, mid_channels, num_classes)
        self.init_weight()

    def forward(self, x):
        dfm = self.db(x)
        feat1, feat2, feat3, feat4, sfm = self.sb(x)
        out = self.head(self.bga(dfm, sfm))
        if not self.training:
            out_list = nn.functional.interpolate(out, x.shape[2:], mode='bilinear', align_corners=True)
        else:
            out_1 = self.aux_head1(feat1)
            out_2 = self.aux_head2(feat2)
            out_3 = self.aux_head3(feat3)
            out_4 = self.aux_head4(feat4)
            out_list = [out, out_1, out_2, out_3, out_4]
        out_list = [nn.functional.interpolate(out, x.shape[2:], mode='bilinear', align_corners=True) for out in out_list]
        return out_list

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

class DetailBranch(nn.Module):
    def __init__(self, stage_channels):
        super(DetailBranch, self).__init__()
        C1, C2, C3 = stage_channels
        self.S1 = nn.Sequential(
            ConvBNRelu(3, C1, 3, stride=2),
            ConvBNRelu(C1, C1, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNRelu(C1, C2, 3, stride=2),
            ConvBNRelu(C2, C2, 3, stride=1),
            ConvBNRelu(C2, C2, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNRelu(C2, C3, 3, stride=2),
            ConvBNRelu(C3, C3, 3, stride=1),
            ConvBNRelu(C3, C3, 3, stride=1),
        )

    def forward(self, x):
        y = self.S1(x)
        y = self.S2(y)
        y = self.S3(y)
        return y


if __name__ == "__main__":
    import torchsummary
    import torchvision.models
    mdl = DetailBranch(64, 64, 128)
    mdl = mdl.cuda()
    torchsummary.summary(mdl, (3, 224, 224))

class SegmenticBranch(nn.Module):
    def __init__(self, stage_channel):
        super(SegmenticBranch, self).__init__()
        C2, C3, C4, C5 = stage_channel
        self.stem = StemBlock(C2)
        self.stage3 = nn.Sequential(
            GatherAndExpansion2(C3,C3),
            GatherAndExpansion(C3)
        )
        self.stage4 = nn.Sequential(
            GatherAndExpansion2(C4, C4),
            GatherAndExpansion(C4)
        )
        self.stage5 = nn.Sequential(
            GatherAndExpansion2(C5,C5),
            GatherAndExpansion(C5)
        )
        self.contextEmBlock = ContextEmdbedBlock(C5)

    def forward(self, x):
        y_4 = self.stem(x)
        y_8 = self.stage3(y_4)
        y_16 = self.stage4(y_8)
        y_32 = self.stage5(y_16)
        out_32 = self.contextEmBlock(y_32)
        return y_4, y_8, y_16, y_32, out_32

class ContextEmdbedBlock(nn.Module):
    def __init__(self, in_channels):
        super(ContextEmdbedBlock, self).__init__()
        self.convAvgPool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(num_features=in_channels),
            ConvBNRelu(in_channels=in_channels, out_channels=in_channels, bias=False, kernel_size=1),
        )
        self.conv_out = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.convAvgPool(x) + x
        y = self.conv_out(y)
        return y


class StemBlock(nn.Module):
    def __init__(self, out_channels):
        super(StemBlock, self).__init__()
        self.conv = ConvBNRelu(in_channels=3, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_l = nn.Sequential(
            ConvBNRelu(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            ConvBNRelu(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        )
        self.mxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBNRelu(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        y = self.conv(x)
        y_l = self.conv_l(y)
        y_r = self.mxPool(y)
        y = torch.cat((y_l, y_r), dim=1)
        y = self.conv_out(y)
        return y



class GatherAndExpansion2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatherAndExpansion2, self).__init__()
        expand_ch = 6 * in_channels
        self.conv = nn.Sequential(
            ConvBNRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            ConvBN(in_channels=in_channels, out_channels=expand_ch, kernel_size=3, padding=1, stride=2, groups=in_channels, bias=False),
            ConvBN(in_channels=expand_ch, out_channels=expand_ch, kernel_size=1, padding=0, groups=expand_ch, bias=False),
            ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1, bias=False)
        )
        self.conv2 = nn.Sequential(
            ConvBN(in_channels=in_channels, out_channels=in_channels, stride=2, padding=1),
            ConvBN(in_channels=in_channels, out_channels=expand_ch, kernel_size=1, padding=0, stride=1, bias=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x) + self.conv2(x)
        y = self.relu(y)
        return y


class GatherAndExpansion(nn.Module):
    def __init__(self, in_channels):
        super(GatherAndExpansion, self).__init__()
        expand_ch = 6 * in_channels
        self.conv = nn.Sequential(
            ConvBNRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            ConvBN(in_channels=in_channels, out_channels=expand_ch, kernel_size=3, padding=1, stride=1, groups=in_channels, bias=False),
            ConvBN(in_channels=expand_ch, out_channels=in_channels, kernel_size=1, padding=0, bias=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x) + x
        y = self.relu(y)
        return y

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



class SegHead(nn.Module):
    def __init__(self, in_dim, mid_dim, num_classes):
        super().__init__()

        self.conv1 = ConvBNRelu(in_channels=in_dim, out_channels=mid_dim, kernel_size=3, padding=0, bias=True, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(in_channels=mid_dim, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return y