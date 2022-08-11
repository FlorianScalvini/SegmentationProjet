from models.backbone import Backbone
import torch.nn as nn
from models.module import ConvBNRelu, ConvAct
_config = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}

_config = {
    "vgg11" : [1, 1, 2, 2, 2],
    "vgg13" : [2, 2, 2, 2, 2],
    "vgg16": [2, 2, 3, 3, 3],
    "vgg19": [2, 2, 4, 4, 4],
}

class VGG(Backbone):

    def __init__(self, type="vgg16", batch_norm=True, logits_out=None, dropout=0.5, num_classes=1000, pretrained=None):
        super(VGG, self).__init__(num_classes=num_classes)
        if not type in _config.keys():
            raise ValueError("Unknown VGG model")
        config = _config[type]
        if logits_out is None:
            self.logits_out = len(config)
        else:
             self.logits_out = logits_out

        self.bn = batch_norm
        self.stage0 = self._make_layers(config[0], 3, 64, batch_norm)
        self.stage1 = self._make_layers(config[1], 64, 128, batch_norm)
        self.stage2 = self._make_layers(config[2], 128, 256, batch_norm)
        self.stage3 = self._make_layers(config[3], 256, 512, batch_norm)
        self.stage4 = self._make_layers(config[4], 512, 512, batch_norm)

        self.Classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(1),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        self._init_weight(pretrained)

    def forward(self, x):
        feature_2 = self.stage0(x)
        feature_4 = self.stage1(feature_2)
        feature_8 = self.stage2(feature_4)
        feature_16 = self.stage3(feature_8)
        feature_32 = self.stage4(feature_16)
        if self._backbone:
            return feature_4, feature_8, feature_16, feature_32
        else:
            return self.Classifier(feature_32)



    @staticmethod
    def _make_layers(self, k, in_channels, out_channels, batch_norm=True):
        block = []
        for i in range(k):
            if batch_norm:
                block.append(ConvBNRelu(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=3, padding=1, bias=True))
            else:
                block.append(ConvAct(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=3, padding=1, bias=True))
        block.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*block)