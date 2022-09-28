import models.BaseModel as BaseModel
from models.backbone import *
import torch.nn as nn
import models.module
import models


class PPLiteSeg(BaseModel):
    def __init__(self, num_classes, backbone="STDC2", arm_type='UAFM_SpAtten', cm_out_ch=128, resize_mode='bilinear', *args, **kwargs):
        super(PPLiteSeg, self).__init__(num_classes=num_classes)
        # backbone
        if backbone == "STDC2":
            self.backbone = STDC2()
            seg_head_inter_chs = [64, 64, 64]
            arm_out_chs = [64, 96, 128]
            cm_bin_sizes = [1, 2, 4]
            backbone_indices = [2, 3, 4]
        else:
            raise NotImplementedError

        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [self.backbone.feat_channels[i] for i in backbone_indices]

        # head
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
            "arm_out_chs and backbone_indices should be equal"

        self.ppseg_head = PPLiteSegHead(backbone_out_chs, arm_out_chs,cm_bin_sizes, cm_out_ch, arm_type, resize_mode)

        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        assert len(seg_head_inter_chs) == len(backbone_indices), "The length of " \
            "seg_head_inter_chs and backbone_indices should be equal"
        self.seg_heads = nn.ModuleList()  # [..., head_16, head32]
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))
        self.init_weight()

    def forward(self, x):
        x_hw = x.shape[2:]

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]

        feats_head = self.ppseg_head(feats_selected)  # [..., x8, x16, x32]

        if self.training:
            logit_list = []
            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)
            logit_list = [
                nn.functional.interpolate(
                    x, x_hw, mode='bilinear', align_corners=False)
                for x in logit_list]
        else:
            x = self.seg_heads[0](feats_head[0])
            logit_list = nn.functional.interpolate(x, x_hw, mode='bilinear', align_corners=False)

        return logit_list

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class PPLiteSegHead(nn.Module):

    def __init__(self, in_channels, uarm_channels, cm_out_channel, cm_out_ch, arm_modes="UAFM", resize_mode='bilinear'):
        super().__init__()
        self.sppm = SPPM(in_channels=in_channels[-1], out_channels=cm_out_ch, inter_channels=cm_out_ch, bin_sizes=cm_out_channel)
        self.armList = nn.ModuleList()  # [..., arm8, arm16, arm32]
        for i in range(len(in_channels)):
            low_chs = in_channels[i]
            high_ch = cm_out_ch if i == len(in_channels) - 1 else uarm_channels[i + 1]
            out_ch = uarm_channels[i]
            arm_module = getattr(models.module, arm_modes)
            arm = arm_module(x_ch=low_chs, y_ch=high_ch, out_ch=out_ch, ksize=3, resize_mode=resize_mode)
            self.armList.append(arm)

    def forward(self, x):
        high_feat = self.sppm(x[-1])
        out_feat_list = []
        for i in reversed(range(len(x))):
            low_feat = x[i]
            arm = self.armList[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)
        return out_feat_list


class SPPM(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, bin_sizes=None, align_corners=False):
        super(SPPM, self).__init__()
        if bin_sizes is None:
            bin_sizes = [1, 2, 4]
        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])
        self.conv = ConvBNRelu(in_channels=inter_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = ConvBNRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        return nn.Sequential(prior, conv)

    def forward(self, x):
        out = None
        input_shape = x.shape[2:]
        for stage in self.stages:
            y = stage(x)
            y = nn.functional.interpolate(y, input_shape, mode="bilinear", align_corners=self.align_corners)
            if out is None:
                out = y
            else:
                out += y
        out = self.conv(out)
        return out

class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super(SegHead, self).__init__()
        self.conv = ConvBNRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)