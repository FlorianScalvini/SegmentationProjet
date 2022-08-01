import torch.nn as nn
from models.module.convBnRelu import ConvBNActivation, ConvBN
from models.module.SqueezeExcitation import SqueezeExcitation
from models.backbone.Backbone import *

class MobilenetV3(Backbone):
    def __init__(self, version=None, num_classes=1000):
        super(MobilenetV3, self).__init__(num_classes=num_classes)
        self.classifier = None
        if version == "small":
            self._smallNetwork()
        elif version == "large":
            self._largeNetwork()
        else:
            raise ValueError(self.__name__ + "is implemented with in large or small version")
        self._init_weight()

    def forward(self, x):
        feature_4 = self.stage1(x)
        feature_8 = self.stage2(feature_4)
        feature_16 = self.stage3(feature_8)
        feature_32 = self.stage4(feature_16)
        if self._backbone:
            return feature_4, feature_8, feature_16, feature_32
        else:
            return self.classifier(feature_32)

    def _largeNetwork(self):
        self.stage1 = nn.Sequential(
            ConvBNActivation(in_channels=3, out_channels=16, stride=2, kernel_size=3, padding=1, bias=False,activation=nn.Hardswish()),
            InvertedResidual(in_channels=16, out_channels=16, exp_channels=16, kernel_size=3, strideDW=1, se=False,activation="RE"),
            InvertedResidual(in_channels=16, out_channels=24, exp_channels=64, kernel_size=3, strideDW=2, se=False,activation="RE"),
            InvertedResidual(in_channels=24, out_channels=24, exp_channels=72, kernel_size=3, strideDW=1, se=False,activation="RE"),
        )

        self.stage2 = nn.Sequential(
            InvertedResidual(in_channels=24, out_channels=40, exp_channels=72, kernel_size=5, strideDW=2, se=True, activation="RE"),
            InvertedResidual(in_channels=40, out_channels=40, exp_channels=120, kernel_size=5, strideDW=1, se=True, activation="RE"),
            InvertedResidual(in_channels=40, out_channels=40, exp_channels=120, kernel_size=5, strideDW=1, se=True, activation="RE"),
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(in_channels=40, out_channels=80, exp_channels=240, kernel_size=3, strideDW=2, se=True, activation="HS"),
            InvertedResidual(in_channels=80, out_channels=80, exp_channels=200, kernel_size=3, strideDW=1, se=True, activation="HS"),
            InvertedResidual(in_channels=80, out_channels=80, exp_channels=184, kernel_size=3, strideDW=1, se=True, activation="HS"),
            InvertedResidual(in_channels=80, out_channels=80, exp_channels=184, kernel_size=3, strideDW=1, se=True, activation="HS"),
            InvertedResidual(in_channels=80, out_channels=112, exp_channels=480, kernel_size=3, strideDW=1, se=True, activation="HS"),
            InvertedResidual(in_channels=112, out_channels=112, exp_channels=672, kernel_size=3, strideDW=1, se=True, activation="HS"),
        )

        self.stage4 = nn.Sequential(
            InvertedResidual(in_channels=112, out_channels=160, exp_channels=672, kernel_size=5, strideDW=2, se=True, activation="HS"),
            InvertedResidual(in_channels=160, out_channels=160, exp_channels=960, kernel_size=5, strideDW=1, se=True, activation="HS"),
            InvertedResidual(in_channels=160, out_channels=160, exp_channels=960, kernel_size=5, strideDW=1, se=True, activation="HS"),
        )
        self.classifier = nn.Sequential(
            ConvBNActivation(in_channels=160, out_channels=960, stride=1, kernel_size=1, bias=False, activation=nn.Hardswish()),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Linear(in_features=576, out_features=1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=self.num_classes, bias=False))

    def _smallNetwork(self):
        self.stage1 = nn.Sequential(
            ConvBNActivation(in_channels=3, out_channels=16, stride=2, kernel_size=3, padding=1, bias=False, activation=nn.Hardswish()),
            InvertedResidual(in_channels=16, out_channels=16, exp_channels=16, kernel_size=3, strideDW=2, se=True, activation="RE")
        )

        self.stage2 = nn.Sequential(
            InvertedResidual(in_channels=16, out_channels=24, exp_channels=72, kernel_size=3, strideDW=2, se=False, activation="RE"),
            InvertedResidual(in_channels=24, out_channels=24, exp_channels=88, kernel_size=3, strideDW=1, se=False, activation="RE"),
            InvertedResidual(in_channels=24, out_channels=40, exp_channels=96, kernel_size=5, strideDW=2, se=True, activation="HS")
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(in_channels=40, out_channels=40, exp_channels=240, kernel_size=5, strideDW=1, se=True, activation="HS"),
            InvertedResidual(in_channels=40, out_channels=40, exp_channels=240, kernel_size=5, strideDW=1, se=True, activation="HS"),
            InvertedResidual(in_channels=40, out_channels=48, exp_channels=120, kernel_size=5, strideDW=1, se=True, activation="HS"),
            InvertedResidual(in_channels=48, out_channels=48, exp_channels=144, kernel_size=5, strideDW=1, se=True, activation="HS"),
        )

        self.stage4 =nn.Sequential(
            InvertedResidual(in_channels=48, out_channels=96, exp_channels=288, kernel_size=5, strideDW=1, se=True, activation="HS"),
            InvertedResidual(in_channels=96, out_channels=96, exp_channels=576, kernel_size=5, strideDW=1, se=True, activation="HS"),
            InvertedResidual(in_channels=96, out_channels=96, exp_channels=576, kernel_size=5, strideDW=1, se=True, activation="HS"),
        )

        self.classifier = nn.Sequential(
            ConvBNActivation(in_channels=96, out_channels=576, stride=1, kernel_size=1, bias=False, activation=nn.Hardswish()),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=self.num_classes, bias=True))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, exp_channels=1, kernel_size=3, strideDW=1,se=False, activation="HS"):
        super(InvertedResidual, self).__init__()
        if activation == "HS":
            act = nn.Hardswish()
        elif activation == "RE":
            act = nn.ReLU(inplace=True)
        else:
            raise ValueError(self.__name__ + "implemnted only RE : Relu or HS : Hardswish activation.")
        if strideDW > 2:
            raise ValueError(self.__name__ + "is not implemented with stride upper than 2.")
        self.strideDW = strideDW

        layers = []
        if exp_channels != in_channels:
            layers.append(ConvBNActivation(in_channels=in_channels, out_channels=exp_channels, activation=act, kernel_size=1, stride=1, bias=False))
        layers.append(
            ConvBNActivation(in_channels=exp_channels, out_channels=exp_channels, activation=act,
                             kernel_size=kernel_size, groups=exp_channels, padding=int(kernel_size / 2), stride=self.strideDW, bias=False))
        if se:

            layers.append(SqueezeExcitation(in_channels=exp_channels, squeeze_channels=int(exp_channels * 0.25), scale_activation=nn.Hardswish()))

        layers.append(
            ConvBNActivation(in_channels=exp_channels, out_channels=out_channels, activation=nn.Identity(),
                             kernel_size=1, stride=1, bias=False))
        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        y = self.block(x)
        if self.strideDW == 2:
            y = y + x
        return y

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)





if __name__ == "__main__":
    import torchsummary
    mdl = MobilenetV3(version="small").cuda()
    mdl.classifier()
    torchsummary.summary(mdl, (3, 224, 224))
    print(mdl)