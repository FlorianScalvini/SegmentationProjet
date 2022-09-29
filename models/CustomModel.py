import torch
import torchvision
from models.module import *
from models.BaseModel import BaseModel


class CSPVovnet(nn.Module):
    def __init__(self):
        super(CSPVovnet, self).__init__()


    def forward(self, x):
        return

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

class PAPPM(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(in_channels, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
            BatchNorm(in_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        )

        self.scale_process = nn.Sequential(
            BatchNorm(mid_channels * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 4, mid_channels * 4, kernel_size=3, padding=1, groups=4, bias=False),
        )

        self.compression = nn.Sequential(
            BatchNorm(mid_channels * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 5, out_channels, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            BatchNorm(in_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        out = self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)
        return out


class SegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super(SegHead, self).__init__()
        self.module = nn.Sequential(ConvBNRelu(in_channels=in_channels, out_channels=mid_channels, kernel_size=3,
                                               padding=1, stride=1, bias=False),
                                    nn.Conv2d(in_channels=mid_channels, out_channels=num_classes, kernel_size=1,
                                              stride=1, padding=0, bias=False))
    def forward(self, x):
        return self.module(x)



class OSA_module(nn.Module):
    def __init__(
        self, in_channels, stage_ch, out_channels, layer_per_block, identity=False, depthwise=False
    ):

        super(OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_channels
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = ConvBNRelu(in_channels=in_channel, out_channels=stage_ch, kernel_size=1, padding=0, stride=1, bias=False)
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(in_channels=stage_ch, out_channels=stage_ch, kernel_size=3, padding=1, groups=stage_ch, bias=False),
                    ConvBNRelu(in_channels=stage_ch, out_channels=stage_ch, padding=0, kernel_size=1,  bias=False)))
            else:
                self.layers.append(ConvBNRelu(in_channels=in_channel, out_channels=stage_ch, kernel_size=3, groups=1, padding=1, bias=False))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_channels + layer_per_block * stage_ch
        self.concat = ConvBNRelu(in_channels=in_channel, out_channels=out_channels, kernel_size=1, padding=0, stride=1, bias=False)


    def forward(self, x):

        identity_feat = x
        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        if self.identity:
            xt = xt + identity_feat

        return xt




class OSAStage(nn.Module):
    def __init__( self, in_channels, stage_ch, concat_ch, block_per_stage, layer_per_block, depthwise=False):
        super(OSAStage, self).__init__()
        layers = []
        layers.append(nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                                    OSA_module(in_channels, stage_ch, concat_ch, layer_per_block,depthwise=depthwise)))
        for i in range(block_per_stage - 1):
            layers.append(OSA_module(concat_ch, stage_ch, concat_ch, layer_per_block, identity=True, depthwise=depthwise))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CustomModel(BaseModel):
    def __init__(self, num_classes, planes=64,  ppm_planes=96, head_planes=128):
        super(CustomModel, self).__init__(num_classes=num_classes)
        self.layer_1 = nn.Sequential(ConvBNRelu(in_channels=3, out_channels=planes, kernel_size=3, stride=2, padding=1, bias=False),
                                    ConvBNRelu(in_channels=planes, out_channels=planes, kernel_size=3, stride=2, padding=1, bias=False))

        self.layer_2d = ConvBNRelu(in_channels=planes, out_channels=planes, kernel_size=3, padding=1, stride=2,
                                   bias=False)
        self.layer_3d = ConvBNRelu(in_channels=planes, out_channels=planes, kernel_size=3, padding=1, stride=1,
                                   bias=False)
        self.layer_4d = ConvBNRelu(in_channels=planes, out_channels=planes*2, kernel_size=3, padding=1, stride=1,
                                   bias=False)

        self.layer_2 = OSAStage(in_channels=64, stage_ch=64, concat_ch=128, block_per_stage=2, layer_per_block=3, depthwise=False)
        self.layer_3 = OSAStage(in_channels=128, stage_ch=80, concat_ch=256, block_per_stage=2, layer_per_block=3, depthwise=False)
        self.layer_4 = OSAStage(in_channels=256, stage_ch=96, concat_ch=384, block_per_stage=2, layer_per_block=3, depthwise=False)
        self.layer_5 = OSAStage(in_channels=384, stage_ch=112, concat_ch=512, block_per_stage=3, layer_per_block=3, depthwise=False)
        self.layer_6 = OSAStage(in_channels=512, stage_ch=128, concat_ch=planes*8, block_per_stage=3, layer_per_block=3, depthwise=False)

        self.pappm = PAPPM(in_channels=planes*8, mid_channels=ppm_planes, out_channels=planes*2)
        self.iaff = iAFF(in_channels=128)
        self.seghead = SegHead(in_channels=128, mid_channels=head_planes, num_classes=num_classes)
        self.seghead_d = SegHead(in_channels=128, mid_channels=head_planes, num_classes=num_classes)

    def forward(self, x):
        y = self.layer_1(x)
        y_c = self.layer_2(y)
        y_c = self.layer_3(y_c)
        y_c = self.layer_4(y_c)
        y_c = self.layer_5(y_c)
        y_c = self.layer_6(y_c)
        y_c = self.pappm(y_c)

        y_d = self.layer_2d(y)
        y_d = self.layer_3d(y_d)
        y_d = self.layer_4d(y_d)
        y_c = nn.functional.interpolate(y_c, y_d.shape[-2:], mode="bilinear", align_corners=False)
        y = self.iaff(y_c, y_d)
        y = self.seghead(y)
        y = nn.functional.interpolate(y, x.shape[-2:], mode="bilinear", align_corners=False)
        if self.training:
            y_d = self.seghead_d(y_d)
            y_d = nn.functional.interpolate(y_d, x.shape[-2:], mode="bilinear", align_corners=False)
            return (y, y_d)
        return y


if __name__ == "__main__":
    import torchsummary
    net = CustomModel(num_classes=19).cuda().eval()
    import numpy as np
    dummy_input = torch.randn(1, 3, 1024, 512, dtype=torch.float).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 3000
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