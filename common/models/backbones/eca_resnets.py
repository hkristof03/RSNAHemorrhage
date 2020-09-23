#  MIT License
#
#  Copyright (c) 2019 Peter Pesti <pestipeti@gmail.com>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from common.models.layers import ECALayer

__all__ = ['EcaResnet', 'eca_resnet18', 'eca_resnet34', 'eca_resnet50', 'eca_resnet101',
           'eca_resnet152']

model_urls = {
    'eca_resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'eca_resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'eca_resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'eca_resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'eca_resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def Norm2d(planes):
    return nn.BatchNorm2d(planes)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, k_size=3):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = Norm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, dilation=dilation)
        self.bn2 = Norm2d(planes)
        self.eca = ECALayer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, k_size=3):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, dilation=dilation)
        self.bn2 = Norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = Norm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.eca = ECALayer(planes * 4, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class EcaResnet(nn.Module):

    def __init__(self, block, layers, use_dilation=False, k_size=(3, 3, 3, 3)):
        if use_dilation:
            last_stride = 1
            dilation = 2
        else:
            last_stride = 2
            dilation = 1

        self.out_channels = 2048
        self.inplanes = 64
        super(EcaResnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = Norm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], k_size=int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, k_size=int(k_size[1]))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, k_size=int(k_size[2]))
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=last_stride, dilation=dilation, k_size=int(k_size[3]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, k_size=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                Norm2d(planes * block.expansion),
                nn.AvgPool2d(stride, stride),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            downsample=downsample, dilation=dilation, k_size=k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def eca_resnet18(pretrained=False, k_size=(3, 3, 3, 3), **kwargs):
    """Constructs a EcaResnet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        k_size (list): Kernel sizes
    """
    model = EcaResnet(BasicBlock, [2, 2, 2, 2], k_size=k_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['eca_resnet18']), strict=False)
    return model


def eca_resnet34(pretrained=False, k_size=(3, 3, 3, 3), **kwargs):
    """Constructs a EcaResnet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        k_size (list): Kernel sizes
    """
    model = EcaResnet(BasicBlock, [3, 4, 6, 3], k_size=k_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['eca_resnet34']), strict=False)
    return model


def dilated_eca_resnet34(pretrained=False, k_size=(3, 3, 3, 3), **kwargs):
    """Constructs a EcaResnet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        k_size (list): Kernel sizes
    """
    model = EcaResnet(BasicBlock, [3, 4, 6, 3], k_size=k_size, use_dilation=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['eca_resnet34']), strict=False)
    return model


def dilated_eca_resnet50(pretrained=False, k_size=(3, 3, 3, 3), **kwargs):
    """Constructs a EcaResnet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        k_size (list): Kernel sizes
    """
    model = EcaResnet(Bottleneck, [3, 4, 6, 3], use_dilation=True, k_size=k_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['eca_resnet50']), strict=False)
    return model


def eca_resnet50(pretrained=False, k_size=(3, 3, 3, 3), **kwargs):
    """Constructs a EcaResnet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        k_size (list): Kernel sizes
    """
    model = EcaResnet(Bottleneck, [3, 4, 6, 3], k_size=k_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['eca_resnet50']), strict=False)
    return model


def eca_resnet101(pretrained=False, k_size=(3, 3, 3, 3), **kwargs):
    """Constructs a EcaResnet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        k_size (list): Kernel sizes
    """
    model = EcaResnet(Bottleneck, [3, 4, 23, 3], k_size=k_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['eca_resnet101']), strict=False)
    return model


def eca_resnet152(pretrained=False, k_size=(3, 3, 3, 3), **kwargs):
    """Constructs a EcaResnet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        k_size (list): Kernel sizes
    """
    model = EcaResnet(Bottleneck, [3, 8, 36, 3], k_size=k_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['eca_resnet152']), strict=False)
    return model
