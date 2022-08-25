import torch
import torch.nn as nn
import torch.nn.functional
from models.BaseModel import BaseModel
from models.module import *
import models.backbone


import torch
import torch.nn as nn
import torch.nn.functional
from models.BaseModel import BaseModel
from models.module import *
from models.bisenetv2 import StemBlock
from models.backbone.EfficientNet import MBConvBlock
from utils.utils import make_divisible
import math



class MS_CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(MS_CAM, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNRelu(in_channels=in_channels, out_channels=int(in_channels/reduction_ratio), kernel_size=1),
            ConvBN(in_channels=int(in_channels/reduction_ratio), out_channels=in_channels, kernel_size=1)
        )
        self.branch_2 = nn.Sequential(
            ConvBNRelu(in_channels=in_channels, out_channels=int(in_channels / reduction_ratio), kernel_size=1),
            ConvBN(in_channels=int(in_channels / reduction_ratio), out_channels=in_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.branch_1(x)
        y2 = self.branch_2(x)
        y = y1 + y2
        y = self.sigmoid(y)
        y = y * x
        return y

class AFF(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(AFF, self).__init__()
        self.ms_cam = MS_CAM(in_channels=in_channels, reduction_ratio=reduction_ratio)

    def forward(self, x, x1):
        y = x + x1
        y_l = x * y
        y_r = (1 - y) * x1
        return y_l + y_r


class iAFF(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(iAFF, self).__init__()
        self.aff = AFF(in_channels=in_channels, reduction_ratio=reduction_ratio)
        self.ms_cam = MS_CAM(in_channels=in_channels, reduction_ratio=reduction_ratio)

    def forward(self, x, x1):
        y = self.aff(x, x1)
        y_l = y * x
        y_r = (1-y) * x1
        return y_l + y_r

class eSEModule(nn.Module):
    def __init__(self, in_channel):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, padding=0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        y = self.act(y) * y
        return y


class OneShotAggregation(nn.Module):
    def __init__(self, in_channel, stage_channel, concat_channel, num_layer, identity=False, SE=False, depthwise=False):
        super(OneShotAggregation, self).__init__()
        self.identity = identity
        self.isReduced = False
        self.depthwise = depthwise
        layers = []
        self.SE = False
        in_ch = in_channel

        if self.depthwise and in_channel != stage_channel:
            self.isReduced = True
            self.conv_reduction = ConvBNRelu(in_channels=in_ch, out_channels=stage_channel, kernel_size=1, stride=1)
            in_ch = stage_channel

        for i in range(num_layer):
            if self.depthwise:
                layers.append(ConvBNRelu(in_channels=in_ch, out_channels=stage_channel, kernel_size=3, stride=1, padding=1, groups=in_ch))
            else:
                layers.append(ConvBNRelu(in_channels=in_ch, out_channels=stage_channel, kernel_size=3, stride=1, padding=1))
            in_ch = stage_channel
        self.layers = nn.Sequential(*layers)
        in_ch = in_channel + num_layer*stage_channel
        self.conv_cat = ConvBNRelu(in_channels=in_ch, out_channels=concat_channel, kernel_size=1, padding=0, stride=1)
        if self.SE:
            self.ese = eSEModule(in_channel)

    def forward(self, x):
        y = x
        output = [x]
        for layer in self.layers:
            y = layer(y)
            output.append(y)
        y = torch.cat(output, dim=1)
        y = self.conv_cat(y)
        if self.SE:
            y = self.ese(y)
        if self.identity:
            y = y + x
        return y


vovnet_19 = {
    'stem' : [64, 64, 128],
    'osa' : [
        [64, 112, 3, 1, True, True, False],
        [80, 256, 3, 1, True, True, False],
        [96, 384, 3, 1, True, True, False],
        [112, 512, 3, 1, True, True, False],
    ]
}

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.aff = AFF(in_channels=in_channels)
        self.conv = ConvBNRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3)

    def forward(self, x_s, x_d):
        y = self.aff(x_s, x_d)
        y = self.conv(y)
        return y


class ContextPath(nn.Module):
    def __init__(self, in_channel, out_channels, config_vovnet=None):
        super(ContextPath, self).__init__()
        if config_vovnet is None:
            config_vovnet = vovnet_19
        config = config_vovnet["osa"]
        C4, C5 = config[-2][1], config[-1][1]
        self.backbone = VoVNet(in_channels=in_channel, config=vovnet_19, idx=[0, 2, 3])

        self.armStage4 = AttentionRefinementModule(in_channels=C4, out_channels=out_channels)
        self.armStage5 = AttentionRefinementModule(in_channels=C5, out_channels=out_channels)
        self.convStage5 = ConvBNRelu(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNRelu(C5, out_channels, 1, bias=False))


    def forward(self, x_2):
        if self.training:
            feat_4, feat_8, feat_16, feat_32 = self.backbone(x_2)
        else:
            _, _, feat_16, feat_32 = self.backbone(x_2)

        avg = self.global_context(feat_32)
        feat_32 = self.armStage5(feat_32)
        feat_32 = feat_32 + avg
        feat_32_1 = nn.functional.interpolate(feat_32, scale_factor=2, mode='nearest')
        feat_32_1 = self.convStage5(feat_32_1)
        feat_16 = self.armStage4(feat_16)
        feat_16 = feat_16 + feat_32_1


        if self.training:
            return feat_4, feat_8, feat_16, feat_32
        else:
            return feat_16



class SpatialPath(nn.Module):
    def __init__(self, C1, C2, C3):
        super(SpatialPath, self).__init__()
        self.conv_1c = nn.Sequential(
            ConvBNRelu(in_channels=3, out_channels=C1, kernel_size=3, stride=2, padding=1),
            ConvBNRelu(in_channels=C1, out_channels=C1, kernel_size=3, padding=1)
        )
        self.conv_2c = nn.Sequential(
            ConvBNRelu(in_channels=C1, out_channels=C2, kernel_size=3, stride=2, padding=1),
            ConvBNRelu(in_channels=C2, out_channels=C2, kernel_size=3, padding=1)
        )
        self.conv_3c = nn.Sequential(
            ConvBNRelu(in_channels=C2, out_channels=C3, kernel_size=3, stride=2, padding=1),
            ConvBNRelu(in_channels=C3, out_channels=C3, kernel_size=3, padding=1)
        )

        self.conv_1d = nn.Sequential(
            ConvBNRelu(in_channels=1, out_channels=C1, kernel_size=3, stride=2, padding=1),
            ConvBNRelu(in_channels=C1, out_channels=C1, kernel_size=3, padding=1)
        )
        self.conv_2d = nn.Sequential(
            ConvBNRelu(in_channels=C1, out_channels=C2, kernel_size=3, stride=2, padding=1),
            ConvBNRelu(in_channels=C2, out_channels=C2, kernel_size=3, padding=1)
        )
        self.conv_3d = nn.Sequential(
            ConvBNRelu(in_channels=C2, out_channels=C3, kernel_size=3, stride=2, padding=1),
            ConvBNRelu(in_channels=C3, out_channels=C3, kernel_size=3, padding=1)
        )

        self.sam_1 = SpatialAttention()
        self.sam_2 = SpatialAttention()
        self.sam_3 = SpatialAttention()

        self.conv1 = ConvBNRelu(C1, C1, kernel_size=1)
        self.conv2 = ConvBNRelu(C2, C2, kernel_size=1)
        self.conv3 = ConvBNRelu(C3, C3, kernel_size=1)

    def forward(self, x, x1):
        y_l1 = self.conv_1c(x)
        y_r = self.conv_1d(x1)
        y_2 = self.conv1(y_r + y_l1)
        y_2 = self.sam_1(y_2) * y_2

        y_l = self.conv_2c(y_l1)
        y_r = self.conv_2d(y_r)
        y_4 = self.conv2(y_r + y_l)
        y_4 = self.sam_2(y_4) * y_4

        y_l = self.conv_3c(y_l)
        y_r = self.conv_3d(y_r)
        y_8 = self.conv3(y_r + y_l)
        y_8 = self.sam_3(y_8) * y_8

        return y_l1, y_2, y_4, y_8


class AFS(nn.Module):
    def __init__(self):
        super(AFS, self).__init__()
        self.sam = SpatialAttention()

    def forward(self, x, x1):
        y = x + x1
        y_l = x * y
        y_r = x * (1-y)
        return y_l + y_r


class DecoderFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DecoderFusionModule, self).__init__()
        self.aff = AFF(in_channels=in_channels)
        self.conv = ConvBNRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x_H, x_L):
        x_H = F.interpolate(x_H, scale_factor=2, mode='bilinear', align_corners=False)
        y = self.aff(x_H, x_L)
        y = self.conv(y)
        return y


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


class CustomModel2(BaseModel):
    def __init__(self, num_classes):
        super(CustomModel2, self).__init__(num_classes=num_classes, depth=True)
        C1, C2, C3 = 32, 64, 128
        self.spatialPath = SpatialPath(C1=C1, C2=C2, C3=C3)
        self.contextPath = ContextPath(in_channel=32, out_channels=128, config_vovnet=vovnet_19)
        self.dfm_8 = DecoderFusionModule(in_channels=128, out_channels=64)
        self.dfm_4 = DecoderFusionModule(in_channels=64, out_channels=32)
        self.dfm_2 = DecoderFusionModule(in_channels=32, out_channels=32)
        self.seghead = SegHead(in_dim=32, mid_dim=32, num_classes=num_classes)
        self.seghead_32 = SegHead(in_dim=128, mid_dim=64, num_classes=num_classes)
        self.seghead_16 = SegHead(in_dim=128, mid_dim=64, num_classes=num_classes)
        self.seghead_8 = SegHead(in_dim=256, mid_dim=64, num_classes=num_classes)

    def forward(self, x, x1):
        y, spat_2, spat_4, spat_8 = self.spatialPath(x, x1)
        if self.training:
            feat_4, feat_8, feat_16, feat_32 = self.contextPath(y)
        else:
            feat_16 = self.contextPath(y)
            feat_4, feat_8, feat_32 = None, None, None
        out = self.dfm_8(feat_16, spat_8)
        out = self.dfm_4(out, spat_4)
        out = self.dfm_2(out, spat_2)
        out = self.seghead(out)
        if self.training:
            out = [out]
            out.append(self.seghead_32(feat_32))
            out.append(self.seghead_16(feat_16))
            out.append(self.seghead_8(feat_8))
            out = [nn.functional.interpolate(out_feat, x.shape[2:], mode='bilinear', align_corners=True) for out_feat in out]
            return out
        else:
            out = nn.functional.interpolate(out, x.shape[2:], mode='bilinear', align_corners=True)
        return out


class VoVNet(nn.Module):
    def __init__(self, in_channels, config, idx=None):
        super(VoVNet, self).__init__()
        osa_layer = config['osa']
        layers = []
        in_ch = in_channels
        stage, out_channel, n, k, eSE, identity, dw = config['osa'][0]
        self.stage2 = self._makeLayer(in_channel=in_ch, stage_channel=stage, concat_channel=out_channel, num_layer=n,
                                         depthwise=dw, num_block=k, identity=identity, SE=eSE)
        in_ch = out_channel
        stage, out_channel, n, k, eSE, identity, dw = config['osa'][1]
        self.stage3 = self._makeLayer(in_channel=in_ch, stage_channel=stage, concat_channel=out_channel, num_layer=n,
                                         depthwise=dw, num_block=k, identity=identity, SE=eSE)
        in_ch = out_channel
        stage, out_channel, n, k, eSE, identity, dw = config['osa'][2]
        self.stage4 = self._makeLayer(in_channel=in_ch, stage_channel=stage, concat_channel=out_channel, num_layer=n,
                                         depthwise=dw, num_block=k, identity=identity, SE=eSE)
        in_ch = out_channel
        stage, out_channel, n, k, eSE, identity, dw = config['osa'][3]
        self.stage5 = self._makeLayer(in_channel=in_ch, stage_channel=stage, concat_channel=out_channel, num_layer=n,
                                         depthwise=dw, num_block=k, identity=identity, SE=eSE)
        if idx is None:
            self.idx = [0,1,2,3]
        else:
            self.idx = idx
        self.layers = layers

    def forward(self, x):
        y4 = self.stage2(x)
        y8 = self.stage3(y4)
        y16 = self.stage4(y8)
        y32 = self.stage5(y16)
        return y4, y8, y16, y32

    def _makeLayer(self, in_channel, stage_channel, concat_channel, identity, num_block, num_layer, depthwise, SE):
        layers = [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
        se = SE
        if num_block == 1 and SE:
            se = False
        layers.append(OneShotAggregation(in_channel=in_channel , stage_channel=stage_channel,
                                     concat_channel=concat_channel, num_layer=num_layer, depthwise=depthwise, SE=se))
        for i in range(num_block):
            if SE and i == num_block - 2:
                se = True
            else:
                se = False
            layers.append(
                OneShotAggregation(in_channel=concat_channel, stage_channel=stage_channel, concat_channel=concat_channel,
                                   num_layer=num_layer, depthwise=depthwise, SE=se, identity=identity))
        return nn.Sequential(*layers)


class Stem(nn.Module):
    def __init__(self, lst_channel):
        super(Stem, self).__init__()
        layer = []
        in_ch = lst_channel[0]
        layer.append(ConvBNRelu(in_channels=3, out_channels=in_ch, kernel_size=3, stride=2, padding=1))

        for channel in lst_channel[1:]:
            layer.append(ConvBNRelu(in_channels=in_ch, out_channels=channel, kernel_size=3, stride=1, padding=1))
            in_ch = channel
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        y = self.layer(x)
        return y


if __name__ == "__main__":
    import torchsummary
    net = CustomModel2(num_classes=19).cuda().eval()
    import numpy as np
    dummy_input = torch.randn(1, 3, 1024, 512, dtype=torch.float).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = net(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = net(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time


    torchsummary.summary(net, (3, 1024, 512))
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)

