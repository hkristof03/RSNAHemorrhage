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
import timm

from .densenets import densenet121, densenet161, densenet169, densenet201
from .resnets import resnet34, resnet101, resnet18, resnet50, resnet152
from .efficientnets import EfficientNet
from .inceptions import inception_v3, InceptionV4
from pretrainedmodels import se_resnext50_32x4d

def convert_to_inplace_relu(model):
    # make all relus inplace: https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/7  # noqa
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU(inplace=True))
        else:
            convert_to_inplace_relu(child)


class NoOp(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


def create_classifier_backbone(encoder_slug, pretrained=False, num_classes=1000):

    # ============
    # RESNETS
    if encoder_slug == 'resnet-18':
        encoder = resnet18(pretrained=pretrained)
        num_bottleneck_filters = 512
    elif encoder_slug == 'resnet-34':
        encoder = resnet34(pretrained=pretrained)
        num_bottleneck_filters = 512
    elif encoder_slug == 'resnet-50':
        encoder = resnet50(pretrained=pretrained)
        num_bottleneck_filters = 2048
    elif encoder_slug == 'resnet-101':
        encoder = resnet101(pretrained=pretrained)
        num_bottleneck_filters = 2048
    elif encoder_slug == 'resnet-152':
        encoder = resnet152(pretrained=pretrained)
        num_bottleneck_filters = 2048

    elif encoder_slug == 'resnet-50d':
        encoder = timm.create_model('gluon_resnet50_v1d', pretrained=pretrained, num_classes=2)
        convert_to_inplace_relu(encoder)

        encoder.global_pool = NoOp()
        encoder.flat = NoOp()
        encoder.drop = None
        encoder.output = NoOp()
        encoder.fc = NoOp()

        num_bottleneck_filters = 2048
    elif encoder_slug == 'resnet-101d':
        encoder = timm.create_model('gluon_resnet101_v1d', pretrained=pretrained)
        convert_to_inplace_relu(encoder)

        encoder.global_pool = NoOp()
        encoder.flat = NoOp()
        encoder.drop = None
        encoder.output = NoOp()
        encoder.fc = NoOp()

        num_bottleneck_filters = 2048

    # ============
    # SE-RESNEXT
    elif encoder_slug == 'seresnext-50.32.4d':
        # encoder = timm.create_model('gluon_seresnext50_32x4d', pretrained=pretrained, num_classes=num_classes)
        encoder = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet' if pretrained else None)
        encoder.avg_pool = nn.AdaptiveAvgPool2d(1)
        encoder.last_linear = nn.Linear(
            encoder.last_linear.in_features,
            num_classes,
        )
        num_bottleneck_filters = 2048
    elif encoder_slug == 'seresnext-101.32.4d':
        encoder = timm.create_model('gluon_seresnext101_32x4d', pretrained=pretrained, num_classes=num_classes)
        num_bottleneck_filters = 2048

    # ============
    # DENSENETS
    elif encoder_slug == 'densenet-121':
        encoder = densenet121(pretrained=pretrained)
        num_bottleneck_filters = 1024
    elif encoder_slug == 'densenet-161':
        encoder = densenet161(pretrained=pretrained)
        num_bottleneck_filters = 2208
    elif encoder_slug == 'densenet-169':
        encoder = densenet169(pretrained=pretrained)
        num_bottleneck_filters = 1664
    elif encoder_slug == 'densenet-201':
        encoder = densenet201(pretrained=pretrained)
        num_bottleneck_filters = 1920

    # ==============
    # EFFICIENTNETS
    elif encoder_slug == 'efficientnet-b0':
        encoder = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        num_bottleneck_filters = 1280
    elif encoder_slug == 'efficientnet-b1':
        encoder = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)
        num_bottleneck_filters = 1280
    elif encoder_slug == 'efficientnet-b2':
        encoder = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes)
        num_bottleneck_filters = 1408
    elif encoder_slug == 'efficientnet-b3':
        encoder = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
        num_bottleneck_filters = 1536
    elif encoder_slug == 'efficientnet-b4':
        encoder = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        num_bottleneck_filters = 1792
    elif encoder_slug == 'efficientnet-b5':
        encoder = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes)
        num_bottleneck_filters = 2048
    elif encoder_slug == 'efficientnet-b6':
        encoder = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
        num_bottleneck_filters = 2304
    elif encoder_slug == 'efficientnet-b7':
        encoder = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
        num_bottleneck_filters = 2560

    elif encoder_slug == 'inception-v3':
        encoder = inception_v3(pretrained=pretrained, aux_logits=False)
        num_bottleneck_filters = 2048

    elif encoder_slug == 'inception-v4':
        encoder = InceptionV4(num_classes=0)
        num_bottleneck_filters = 1536

    else:
        raise NotImplementedError

    return encoder, num_bottleneck_filters
