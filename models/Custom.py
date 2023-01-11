import torch
import torchvision
from models.module import *
from models.BaseModel import BaseModel
from models.module import SqueezeExcitation
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
    def __init__(self, in_channels, stage_ch, out_channels,
                 layer_per_block=3, kernel_size=3, stride=1, downsample=False):

        super(OSA_module, self).__init__()
        self.downsample = downsample
        self.layers = nn.ModuleList(
            [
                ConvBNRelu(in_channels=in_channels if r==0 else stage_ch, out_channels=stage_ch,
                           kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
                for r in range(layer_per_block)
            ]
        )
        self.exit_conv = ConvBNRelu(in_channels=in_channels + layer_per_block * stage_ch, out_channels=out_channels,
                                    kernel_size=1)
        self.ese = SqueezeExcitation(in_channels=out_channels, squeeze_channels=out_channels // 2)

    def forward(self, x):
        input = x
        if self.downsample:
            x = nn.functional.max_pool2d(x, 3, stride=2, padding=1)
        features = [x]
        for l in self.layers:
            features.append(l(x))
            x = features[-1]
        x = torch.cat(features, dim=1)
        x = self.exit_conv(x)
        x = self.ese(x)
        # All non-downsampling V2 layers have a residual. They also happen to
        # not change the number of channels.
        if not self.downsample:
            x += input
        return x


class FusionAttentionModule(nn.Module):
    def __init__(self, x_high_channels, x_low_channels, out_channels):
        super(FusionAttentionModule, self).__init__()
        self.conv = ConvBNRelu(in_channels=x_low_channels, out_channels=x_high_channels, kernel_size=3,
                               padding=1, bias=True)
        self.attention = CBAMBlock(channel=x_high_channels)
        self.conv_reduction = ConvBNRelu(in_channels=x_high_channels, out_channels=out_channels,
                                         kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, x_low, x_high):
        x_low = self.conv(x_low)
        x_high = nn.functional.interpolate(x_high, x_low.shape[-2:], mode="bilinear", align_corners=False)
        y = self.attention(x_high + x_low)
        y_high = y * y
        y_low = (1-y) * y
        y = y_low + y_high
        y = self.conv_reduction(y)
        return y

class VovnetBackbone(nn.Module):
    def __init__(self, conf, planes=64):
        super(VovnetBackbone, self).__init__()
        in_ch = 128
        self.stem = nn.Sequential(ConvBNRelu(in_channels=3, out_channels=planes, kernel_size=3, stride=2,
                                                padding=1, bias=False),
                                     ConvBNRelu(in_channels=planes, out_channels=planes, kernel_size=3, stride=1,
                                                padding=1, bias=False),
                                     ConvBNRelu(in_channels=planes, out_channels=in_ch, kernel_size=3, stride=1,
                                                padding=1, bias=False))
        body_layers = []
        for idx, block in enumerate(conf):
            kernel_size, inner_ch, layer_per_block, out_ch, downsample = block
            body_layers.append(OSA_module(
                in_ch,
                inner_ch,
                out_ch,
                layer_per_block=layer_per_block,
                kernel_size=kernel_size,
                downsample=downsample,
            ))
            in_ch = out_ch
        self.body = nn.ModuleList(body_layers)
    def forward(self, x):
        y = self.stem(x)
        for i in range(len(self.body) - 3):
            y = self.body[i](y)
        y_8 = self.body[-3](y)
        y_16 = self.body[-2](y_8)
        y_32 = self.body[-1](y_16)
        return y_32, y_16, y_8


class Custom(BaseModel):
    def __init__(self, num_classes, planes=64,  ppm_planes=96, ppm_out_planes=128, head_planes=128, output_heads=[32, 64, 128]):
        super(Custom, self).__init__(num_classes=num_classes)
        self.conv_x = nn.Sequential(ConvBNRelu(in_channels=3, out_channels=32, stride=2, padding=1),
                                    ConvBNRelu(in_channels=32, out_channels=32, stride=1, padding=1))
        self.conv_0 = nn.Sequential(ConvBNRelu(in_channels=32, out_channels=32, stride=1, padding=1))
        self.conv_1 = nn.Sequential(ConvBNRelu(in_channels=32, out_channels=32, stride=1, padding=1))
        self.conv_2 = nn.Sequential(ConvBNRelu(in_channels=32, out_channels=64, stride=1, padding=1))



    def forward(self, x):
        backbone = self.backbone(x)
        high_feat = self.pappm(backbone[0])
        out_feat_list = []

        for i in range(len(backbone)):
            low_feat = backbone[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)
        if self.training:
            for i in reversed(range(len(out_feat_list))):
                out_feat_list[i] = self.seghead[i](out_feat_list[i])
                out_feat_list[i] = nn.functional.interpolate(out_feat_list[i], x.shape[-2:], mode="bilinear", align_corners=False)
            return out_feat_list
        else:
            y = self.seghead[-1](out_feat_list[-1])
            y = nn.functional.interpolate(y, x.shape[-2:], mode="bilinear", align_corners=False)
            return y


class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio, stride, stoch_depth_prob=0.2):
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        activation_layer = nn.SiLU

        expanded_channels = makeChl(channels=in_channels, expand_ratio=expand_ratio)
        if expanded_channels != in_channels:
            # fused expand
            layers.append(
                ConvBNActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=activation_layer,
                    padding=(kernel_size - 1) // 2,
                )
            )
            # project
            layers.append(
                ConvBNActivation(expanded_channels, out_channels, kernel_size=1, activation=None, padding=0))
        else:
            layers.append(
                ConvBNActivation(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=activation_layer,
                    padding=(kernel_size - 1) // 2
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stoch_depth_prob, "row")
        self.out_channels = out_channels

    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

def makeChl(channels, expand_ratio, min_channels=None):
    if min_channels is None:
        min_channels = channels
    else:
        min_channels = min(channels, min_channels)
    return max(min_channels, int(channels * expand_ratio))

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, stoch_depth_prob=0.2):
        super(MBConvBlock, self).__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = makeChl(in_channels, expand_ratio)
        if expanded_channels != in_channels:
            layers.append(
                ConvBNActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=1,
                    padding=0,
                    activation=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvBNActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=expanded_channels,
                padding=(kernel_size - 1) // 2,
                activation=activation_layer
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, in_channels // 4)
        layers.append(SqueezeExcitation(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            ConvBNActivation(
                expanded_channels, out_channels, kernel_size=1,  padding=0, activation=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stoch_depth_prob, "row")
        self.out_channels = out_channels

    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

if __name__ == "__main__":
    import torchsummary
    net = Custom(num_classes=19).cuda().eval()
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