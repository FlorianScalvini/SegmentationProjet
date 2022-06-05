import torch
import torchvision
import torch.nn as nn
from models.module.convBnRelu import ConvBNRelu, ConvBN
from utils.helpers import load_from_url

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth"
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Resnet(nn.Module):
    def __init__(self, block, layers=None, pretrained=False, num_classes=1000, groups=1, width_per_group=64):
        super(Resnet, self).__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        if len(layers) != 4:
            raise ValueError("Resnet implemented with 4 layers")
        self.expansion = 4
        self.dilation = 1
        self.inplanes = 64
        self.base_width = width_per_group
        self.groups = groups
        self.conv1 = ConvBNRelu(3, self.inplanes, kernel_size=7, padding=3, stride=2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 =  self._make_layer(block=block, planes=64, blocks=layers[0])
        self.layer2 = self._make_layer(block=block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, blocks=layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        if pretrained is not None:
            # Mapping Pytorch Resnet with this Resnet implementation
            state_dict = torch.load(pretrained)
            self.load_state_dict(state_dict)
        else:
            self._init_weight()

    def _init_weight(self, zero_init_residual=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
        layers = [block(
            self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation
        )]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x, logits=False):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups: int = 1,
                 base_width: int = 64, dilation=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64 or dilation > 1:
            raise ValueError("BasicBlock only supports groups=1, dilation = 1 and base_width=64")
        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=out_channels, padding=1, stride=stride, bias=False)
        self.conv2 = ConvBN(in_channels=out_channels, out_channels=out_channels, padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        y = self.relu(y)
        return y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups: int = 1,
                 base_width: int = 64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(out_channels * (base_width / 64.0)) * groups
        norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvBNRelu(in_channels=in_channels, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.conv2 = ConvBNRelu(in_channels=width, out_channels=width, kernel_size=3, stride=stride, groups=groups,
                                padding=dilation, dilation=dilation, bias=False)
        self.conv3 = ConvBN(in_channels=width, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)

        return out

def Resnet18(pretrained=False):
    return Resnet(block=BasicBlock, layers=[2,2,2,2], pretrained=pretrained)

def Resnet34(pretrained=False):
    return Resnet(block=BasicBlock, layers=[3,4,6,3], pretrained=pretrained)

def Resnet50(pretrained=False):
    return Resnet(block=Bottleneck, layers=[3,4,6,3], pretrained=pretrained)

def Resnet101(pretrained=False):
    return Resnet(block=Bottleneck, layers=[3,4,23,3], pretrained=pretrained)

def Resnet152(pretrained=False):
    return Resnet(block=Bottleneck, layers=[3,8,36,3], pretrained=pretrained)


if __name__ == "__main__":
    num = 152
    d = torchvision.models.resnet152(pretrained=True)
    torch.save(d.state_dict(), '/Users/florianscalvini/PycharmProjects/SegmentationProjet/pretrained/resnet/resnet'+ str(num) + '.pth')
    c = Resnet152("/Users/florianscalvini/PycharmProjects/SegmentationProjet/pretrained/resnet/resnet"+ str(num) + ".pth")
    torch.save(c.state_dict(), '/Users/florianscalvini/PycharmProjects/SegmentationProjet/pretrained/resnet/resnet'+ str(num) + '.pth')
    a  = torch.rand((1,3,224,224))
    p = d(a)
    b = c(a)
    print(b[0][0], p[0][0])