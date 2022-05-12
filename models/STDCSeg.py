import torch
import torch.nn as nn
import torch.nn.functional
from models.BaseModel import BaseModel
from models.module import *
import models.backbone


class STDCSeg(BaseModel):
    def __init__(self, num_classes, backbone, pretrained=False,   use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, *args, **kwargs):
        super().__init__(num_classes=num_classes, pretrained=pretrained, backbone=backbone)
        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        self.contextPath = ContextPath(backbone=self.backbone, pretrained=self.pretrained, last_conv=use_conv_last)
        self.ffm = FeatureFusionModule(384, 256)
        self.conv_out = SegHead(256, 256, num_classes)
        self.conv_out8 = SegHead(128, 64, num_classes)
        self.conv_out16 = SegHead(128, 64, num_classes)
        self.conv_out_sp16 = SegHead(512, 64, 1)
        self.conv_out_sp8 = SegHead(256, 64, 1)
        self.conv_out_sp4 = SegHead(64, 64, 1)
        self.conv_out_sp2 = SegHead(32, 64, 1)
        self.pretrained = pretrained
        self.init_weight()


    def forward(self, x):
        x_size = x.size()[2:]
        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.contextPath(x)
        logit_list = []
        if self.training:
            feat_fuse = self.ffm(feat_res8, feat_cp8)
            feat_out = self.conv_out(feat_fuse)
            feat_out8 = self.conv_out8(feat_cp8)
            feat_out16 = self.conv_out16(feat_cp16)

            logit_list = [feat_out, feat_out8, feat_out16]
            logit_list = [ torch.nn.functional.interpolate(x, x_size, mode='bilinear', align_corners=True) for x in logit_list]

            if self.use_boundary_2:
                feat_out_sp2 = self.conv_out_sp2(feat_res2)
                logit_list.append(feat_out_sp2)
            if self.use_boundary_4:
                feat_out_sp4 = self.conv_out_sp4(feat_res4)
                logit_list.append(feat_out_sp4)
            if self.use_boundary_8:
                feat_out_sp8 = self.conv_out_sp8(feat_res8)
                logit_list.append(feat_out_sp8)
        else:
            feat_fuse = self.ffm(feat_res8, feat_cp8)
            feat_out = self.conv_out(feat_fuse)
            feat_out = nn.functional.interpolate(feat_out, x_size, mode='bilinear', align_corners=True)
            logit_list = [feat_out]
        return logit_list

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, SegHead)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class ContextPath(nn.Module):
    def __init__(self, backbone, last_conv=False,pretrained=None, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.backbone = backbone
        self.armStage4 = AttentionRefinementModule(in_channels=512,  out_channels=128)
        in_channels = 1024
        self.armStage5 = AttentionRefinementModule(in_channels=in_channels, out_channels=128)
        self.convStage5 = ConvBNRelu(128, 128, kernel_size=3, padding=1, stride=1)
        self.convStage4 = ConvBNRelu(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv_avg = ConvBNRelu(in_channels, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]
        avg = nn.functional.avg_pool2d(feat32, feat32.size()[2:])

        avg = self.conv_avg(avg)
        avg_up = nn.functional.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.armStage5(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = nn.functional.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.convStage5(feat32_up)

        feat16_arm = self.armStage4(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = nn.functional.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.convStage4(feat16_up)
        return feat2, feat4, feat8, feat16, feat16_up, feat32_up # x8, x16


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

if __name__ == "__main__":
    backbone = models.backbone.STDC(num_classes=19)
    net = STDCSeg(backbone=backbone, num_classes=19)
    net.train()
    in_ten = torch.randn(2, 3, 768, 1536)
    out, out16, out32 = net(in_ten)
    print(out.shape)
