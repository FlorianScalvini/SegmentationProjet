import models.BaseModel as BaseModel
import torch.nn as nn
import torch.nn.functional
from models.module import *
import models
import torch


class PPLiteSeg(BaseModel):
    def __init__(self, num_classes, backbone, backbone_indices=None, arm_type='UAFM_SpAtten',cm_bin_sizes=None, cm_out_ch=128, arm_out_chs=None, seg_head_inter_chs=None, resize_mode='bilinear',pretrained=None, *args, **kwargs):
        super(PPLiteSeg, self).__init__(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
        # backbone
        if seg_head_inter_chs is None:
            seg_head_inter_chs = [64, 64, 64]
        if arm_out_chs is None:
            arm_out_chs = [64, 96, 128]
        if cm_bin_sizes is None:
            cm_bin_sizes = [1, 2, 4]
        if backbone_indices is None:
            backbone_indices = [2, 3, 4]

        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feature_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feature_channels) > max(backbone_indices), \
            f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone_indices) > 1, "The lenght of backbone_indices should be greater than 1"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

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
        x_hw = x(x)[2:].shape

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
        if self.pretrained is not None:
            state_dict = torch.load(self.pretrained)["state_dict"]
            self_state_dict = self.state_dict()
            for k, v in state_dict.items():
                self_state_dict.update({k: v})
            self.load_state_dict(self_state_dict)


class PPLiteSegHead(nn.Module):

    def __init__(self, in_channels, uarm_channels, sppm_out_channel, sppm_sizes, am_type='channel', avgMean=True, resize_mode='bilinear'):
        super().__init__()
        self.sppm = SPPM(in_channels=in_channels[-1], out_channels=sppm_out_channel, inter_channels=sppm_out_channel, bin_sizes=sppm_sizes)
        self.armList = nn.ModuleList()  # [..., arm8, arm16, arm32]
        for i in range(len(in_channels)):
            low_chs = in_channels[i]
            high_ch = sppm_out_channel if i == len(in_channels) - 1 else uarm_channels[i + 1]
            out_ch = uarm_channels[i]
            arm = UAFM(in_channels_Low=low_chs, in_channels_High=high_ch, out_channels=out_ch, kernel_size=3, am_type=am_type, resize_mode=resize_mode, avgMean=avgMean)
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
        input_shape = x[2:].shape
        for stage in self.stages:
            y = stage(input)
            y = nn.functional.interpolate(x, input_shape, mode="bilinear", align_corners=self.align_corners)
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