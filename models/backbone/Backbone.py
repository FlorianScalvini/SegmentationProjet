import torch.nn as nn
import torch
import math

class Backbone(nn.Module):
    def __init__(self, num_classes=1000):
        super(Backbone, self).__init__()
        self._backbone = True
        self.num_classes=1000

    def forward(self, x):
        raise NotImplementedError

    def _init_weight(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)
        if pretrained is not None:
            state_dict = torch.load(pretrained)["state_dict"]
            self_state_dict = self.state_dict()
            for k, v in state_dict.items():
                self_state_dict.update({k: v})
            self.load_state_dict(self_state_dict)

    def __str__(self):
        nbr_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super(Backbone, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'

    def backbone(self):
        self._backbone = True

    def classifier(self):
        self._backbone = False
