import torch.nn as nn
from models.module.convBnRelu import ConvBNActivation, ConvBN

def _make_divisible(v, divisor):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobilenetV2(nn.Module):

    def __init__(self, width_mult=1.0, min_channel=16):
        super(MobilenetV2, self).__init__()
        self.min_channel = min_channel
        self.width_mult = width_mult

        chls = self._makeChl(32)
        self.stage0 = ConvBNActivation(in_channels=3, out_channels=chls, stride=2,
                                       kernel_size=3, padding=1, bias=False, activation=nn.ReLU6(inplace=True))
        in_chls = chls
        chls = self._makeChl(16)
        self.stage1 = InvertedResidual(in_channels=in_chls, out_channels=chls,
                                       strideDW=1, expansion=1)
        in_chls = chls
        chls = self._makeChl(24)
        self.stage2 = nn.Sequential(
            InvertedResidual(in_channels=in_chls, out_channels=chls, strideDW=2, expansion=6),
            InvertedResidual(in_channels=chls, out_channels=chls, strideDW=1, expansion=6)
        )

        in_chls = chls
        chls = self._makeChl(32)
        self.stage3 = nn.Sequential(
            InvertedResidual(in_channels=in_chls, out_channels=chls, strideDW=2, expansion=6),
            InvertedResidual(in_channels=chls, out_channels=chls, strideDW=1, expansion=6),
            InvertedResidual(in_channels=chls, out_channels=chls, strideDW=1, expansion=6)
        )

        in_chls = chls
        chls = self._makeChl(64)
        self.stage4 = nn.Sequential(
            InvertedResidual(in_channels=in_chls, out_channels=chls, strideDW=2, expansion=6),
            InvertedResidual(in_channels=chls, out_channels=chls, strideDW=1, expansion=6),
            InvertedResidual(in_channels=chls, out_channels=chls, strideDW=1, expansion=6),
            InvertedResidual(in_channels=chls, out_channels=chls, strideDW=1, expansion=6)
        )

        in_chls = chls
        chls = self._makeChl(96)
        self.stage5 = nn.Sequential(
            InvertedResidual(in_channels=in_chls, out_channels=chls, strideDW=2, expansion=6),
            InvertedResidual(in_channels=chls, out_channels=chls, strideDW=1, expansion=6),
            InvertedResidual(in_channels=chls, out_channels=chls, strideDW=1, expansion=6)
        )

        in_chls = chls
        chls = self._makeChl(160)
        self.stage6 = nn.Sequential(
            InvertedResidual(in_channels=in_chls, out_channels=chls, strideDW=2, expansion=6),
            InvertedResidual(in_channels=chls, out_channels=chls, strideDW=1, expansion=6),
            InvertedResidual(in_channels=chls, out_channels=chls, strideDW=1, expansion=6)
        )

        in_chls = chls
        chls = self._makeChl(320)
        self.stage7 = nn.Sequential(
            InvertedResidual(in_channels=in_chls, out_channels=chls, strideDW=1, expansion=6)
        )

    def _makeChl(self, channels):
        min_channel = min(channels, self.min_channel)
        return max(min_channel, int(channels * self.width_mult))

    def forward(self, x):
        if
        feat_list = []
        feature_1_2 = self.stage0(x)
        feature_1_2 = self.stage1(feature_1_2)
        feature_1_4 = self.stage2(feature_1_2)
        feature_1_8 = self.stage3(feature_1_4)
        feature_1_16 = self.stage4(feature_1_8)
        feature_1_16 = self.stage5(feature_1_16)
        feature_1_32 = self.stage6(feature_1_16)
        feature_1_32 = self.stage7(feature_1_32)
        feat_list.append(feature_1_4)
        feat_list.append(feature_1_8)
        feat_list.append(feature_1_16)
        feat_list.append(feature_1_32)
        return feat_list


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, strideDW=1, expansion=1):
        super(InvertedResidual, self).__init__()
        if strideDW > 2 :
            raise ValueError(self.__name__ + "is not implemented with stride upper than 2.")
        self.strideDW = strideDW
        self.identity = strideDW == 1 and in_channels == out_channels
        layers = []
        if expansion != 1:
            layers.append(ConvBNActivation(in_channels=in_channels, out_channels=in_channels*expansion, activation=nn.ReLU6(inplace=True), kernel_size=1, stride=1, bias=False))
        layers.append(ConvBNActivation(in_channels=in_channels*expansion, out_channels=in_channels*expansion,
                                       groups=in_channels*expansion, activation=nn.ReLU6(inplace=True), kernel_size=3,
                                       stride=self.strideDW, padding=1, bias=False))
        layers.append(ConvBN(in_channels=in_channels*expansion, out_channels=out_channels, kernel_size=1, stride=1, bias=False))
        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        y = self.block(x)
        if self.identity:
            y = y + x
        return y

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


if __name__ == "__main__":
    import torchvision.models
    mdl = MobilenetV2()
    print(mdl)