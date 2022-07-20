import torch.nn as nn
from models.module.convBnRelu import ConvBNActivation, ConvBN

class MobilenetV2(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobilenetV2, self).__init__()

        layer = []
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        layer.append(ConvBNActivation(in_channels=3, out_channels=32, stride=2, kernel_size=3, padding=1, bias=False, activation=nn.ReLU6(inplace=True)))
        in_channel = 32
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                layer.append(InvertedResidual(in_channels=in_channel, out_channels=c, stride=s, expansion=t))
                in_channel = c
        self.features = nn.Sequential(*layer)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=num_classes, bias=False)
        )

    def forward(self, x):
        y = self.features(x)
        y = self.classifier(y)
        return y




class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1):
        super(InvertedResidual, self).__init__()
        if stride > 2 :
            raise ValueError(self.__name__ + "is not implemented with stride upper than 2.")
        layers = []
        if expansion != 1:
            layers.append(ConvBNActivation(in_channels=in_channels, out_channels=in_channels*expansion, activation=nn.ReLU6(inplace=True), kernel_size=1, stride=1, bias=False))
        layers.append(ConvBNActivation(in_channels=in_channels*expansion, out_channels=in_channels*expansion,
                                       groups=in_channels*expansion, activation=nn.ReLU6(inplace=True), kernel_size=3,
                                       stride=stride, padding=1, bias=False))
        layers.append(ConvBN(in_channels=in_channels*expansion, out_channels=out_channels, kernel_size=3, stride=stride, bias=False))
        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        y = self.block(x)
        if self.stride == 2:
            y = y + x
        return y

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

if __name__ == "__main__":
    mdl = MobilenetV2(1000)
    print(mdl)