import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_channels, activation=nn.ReLU(inplace=True), scale_activation=nn.Sigmoid()):
        super(SqueezeExcitation, self).__init__()
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1, stride=1),
            activation,
            nn.Conv2d(in_channels=squeeze_channels, out_channels=in_channels, kernel_size=1, stride=1),
            scale_activation
        )

    def forward(self,x):
        y = self.scale(x)
        return y * x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)