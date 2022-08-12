import torch
import torch.nn as nn
import torch.nn.functional
from models.BaseModel import BaseModel
from models.module import *
from models.bisenetv2 import StemBlock
from models.backbone.EfficientNet import MBConvBlock
from utils.utils import make_divisible
import math


class CustomModel(nn.Module):
    def __init__(self, num_classes, width_seg=1.0, depth_seg=1.0, conv_out=128):
        super(CustomModel, self).__init__()
        self.num_classes = num_classes
        act_layer = partial(nn.ReLU, True)
        C1, C2, C3, C4, C5, C6 = 16, 24, 40, 80, 112, conv_out
        db_channels = (64, 64, conv_out)

        config_sb = [
            [C1, C2, 3, 6, 2],
            [C2, C3, 5, 6, 2],
            [C3, C4, 3, 6, 3],
            [C4, C5, 5, 6, 3],
        ]

        for idx in range(len(config_sb)):
            config_sb[idx][0] = make_divisible(config_sb[idx][0] * width_seg, 8)
            config_sb[idx][1] = make_divisible(config_sb[idx][1] * width_seg, 8)
            config_sb[idx][-1] = int(math.ceil(config_sb[idx][-1] * depth_seg))



        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNActivation(in_channels=config_sb[-1][1], out_channels=C6, kernel_size=1, bias=False))

        self.db = DetailBranch(channel_stage=db_channels)
        self.sb = SemanticBranch(config=config_sb)
        self.conv1x1 = ConvBNActivation(in_channels=config_sb[-1][1], out_channels=conv_out, kernel_size=1, stride=1,
                                        activation=act_layer)
        self.ffm = FeatureFusionModule(in_channels=conv_out*2, out_channels=C6)
        self.head = SegHead(C6, C6, num_classes)
        self.aux_head1 = SegHead(C2, C2, num_classes)
        self.aux_head2 = SegHead(C3, C3, num_classes)
        self.aux_head3 = SegHead(C4, C4, num_classes)
        self.aux_head4 = SegHead(C5, C5, num_classes)
        return

    def forward(self,  x_color, x_depth):
        dfm = self.db(x_color, x_depth)
        feat_4, feat_8, feat_16, feat_32 = self.sb(x_color)
        out_sb = self.conv1x1(feat_32)
        out_sb = F.interpolate(out_sb, size=dfm.shape[2:], mode='bilinear', align_corners=True)
        out = self.head(self.ffm(dfm, out_sb))
        if not self.training:
            out_list = nn.functional.interpolate(out, x_color.shape[2:], mode='bilinear', align_corners=True)
        else:
            out_1 = self.aux_head1(feat_4)
            out_2 = self.aux_head2(feat_8)
            out_3 = self.aux_head3(feat_16)
            out_4 = self.aux_head4(feat_32)
            out_list = [out, out_1, out_2, out_3, out_4]
            out_list = [nn.functional.interpolate(out, x_color.shape[2:], mode='bilinear', align_corners=True) for out in out_list]
        return out_list


class DetailBranch(nn.Module):
    def __init__(self, channel_stage, act_layer=partial(nn.SiLU, True)):
        super(DetailBranch, self).__init__()
        C1, C2, C3 = channel_stage


        self.b_color = nn.Sequential(
            ConvBNActivation(in_channels=3, out_channels=C1, kernel_size=3, stride=2, padding=1, activation=act_layer),
            ConvBNActivation(in_channels=C1, out_channels=C1, kernel_size=3, stride=1, padding=1, activation=act_layer),
            ConvBNActivation(in_channels=C1, out_channels=C2, kernel_size=3, stride=2, padding=1, activation=act_layer),
            ConvBNActivation(in_channels=C2, out_channels=C2, kernel_size=3, stride=1, padding=1, activation=act_layer),
            ConvBNActivation(in_channels=C2, out_channels=C3, kernel_size=3, stride=2, padding=1, activation=act_layer),
            ConvBNActivation(in_channels=C3, out_channels=C3, kernel_size=3, stride=1, padding=1, activation=act_layer),
        )
        self.b_depth = nn.Sequential(
            ConvBNActivation(in_channels=1, out_channels=C1, kernel_size=3, stride=2, padding=1, activation=act_layer),
            ConvBNActivation(in_channels=C1, out_channels=C1, kernel_size=3, stride=1, padding=1, activation=act_layer),
            ConvBNActivation(in_channels=C1, out_channels=C2, kernel_size=3, stride=2, padding=1, activation=act_layer),
            ConvBNActivation(in_channels=C2, out_channels=C2, kernel_size=3, stride=1, padding=1, activation=act_layer),
            ConvBNActivation(in_channels=C2, out_channels=C3, kernel_size=3, stride=2, padding=1, activation=act_layer),
            ConvBNActivation(in_channels=C3, out_channels=C3, kernel_size=3, stride=1, padding=1, activation=act_layer),
        )

        self.fAtt = FusionSpatialAttentionModule()
        return

    def forward(self, x_color, x_depth):
        y_depth = self.b_depth(x_depth)
        y_color = self.b_color(x_color)
        y = self.fAtt(y_color, y_depth)
        return y



class SemanticBranch(nn.Module):
    def __init__(self, config):
        super(SemanticBranch, self).__init__()
        #self.stem = StemBlock(out_channels=config[0][0])
        id_stage_block = 0
        total_stage_block = 0
        for i in range(len(config)):
            total_stage_block = total_stage_block + config[i][-1]

        self.conv = ConvBNActivation(3, config[0][0], kernel_size=3, stride=2, activation=partial(nn.SiLU, True), padding=1)

        self.stage1, id_stage_block = _make_layer(config[0], id_stage_block=id_stage_block, total_block_len=total_stage_block,
                                   stoch_depth_prob=0.2)
        self.stage2, id_stage_block = _make_layer(config[1], id_stage_block=id_stage_block, total_block_len=total_stage_block,
                                   stoch_depth_prob=0.2)
        self.stage3, id_stage_block = _make_layer(config[2], id_stage_block=id_stage_block, total_block_len=total_stage_block,
                                   stoch_depth_prob=0.2)
        self.stage4, _ = _make_layer(config[3], id_stage_block=id_stage_block, total_block_len=total_stage_block,
                                   stoch_depth_prob=0.2)
        return

    def forward(self, x):
        y_2 = self.conv(x)
        y_4 = self.stage1(y_2)
        y_8 = self.stage2(y_4)
        y_16 = self.stage3(y_8)
        y_32 = self.stage4(y_16)
        return y_4, y_8, y_16, y_32




def _make_layer(config, id_stage_block, total_block_len, stoch_depth_prob=0.2):
    layers = []
    in_channels, out_channels, kernel_size, exp, n = config
    for ni in range(n):
        sd_prob = stoch_depth_prob * float(id_stage_block) / total_block_len
        if ni == 0:
            layers.append(MBConvBlock(in_channels=in_channels, out_channels=out_channels, expand_ratio=exp, stride=2,
                                      kernel_size=kernel_size, stoch_depth_prob=sd_prob))
        else:
            layers.append(MBConvBlock(in_channels=out_channels, out_channels=out_channels, expand_ratio=exp, stride=1,
                                      kernel_size=kernel_size,stoch_depth_prob=sd_prob))
        id_stage_block = id_stage_block + 1
    return nn.Sequential(*layers), id_stage_block


class SegHead(nn.Module):
    def __init__(self, in_dim, mid_dim, num_classes):
        super(SegHead, self).__init__()
        self.conv1 = ConvBNRelu(in_channels=in_dim, out_channels=mid_dim, kernel_size=3, padding=0, bias=True, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(in_channels=mid_dim, out_channels=num_classes, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.init_weight()

    def forward(self, x):
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return y

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
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


class FusionSpatialAttentionModule(nn.Module):
    def __init__(self):
        super(FusionSpatialAttentionModule, self).__init__()
        self.spAtt_1 = SpatialAttention()
        self.spAtt_2 = SpatialAttention()

    def forward(self, x_1, x_2):
        y_1 = self.spAtt_1(x_1)
        y_1 = y_1 * x_1
        y_2 = self.spAtt_2(x_2)
        y_2 = y_2 * x_2
        y = y_1 + y_2
        return y


if __name__ == "__main__":
    import torchsummary
    import torchvision.models
    mdl = CustomModel(num_classes=19, width_seg=1.0, depth_seg=1.0)
    mdl = mdl.cuda().eval()
    torchsummary.summary(mdl, (3, 1024, 512))
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(mdl, (3, 2048, 1024), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

