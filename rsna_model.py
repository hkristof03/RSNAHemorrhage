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
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.models import AbstractModel, create_classifier_backbone, weight_init


def model_factory(args, logger=None, weight_file=None):
    # rsna-resnet-18_b_0
    net_model = args.net_model
    assert isinstance(net_model, str), "--net_model argument is missing"

    arch, encoder, encoder_type = net_model.lower().split('-')
    encoder_slug = encoder + '-' + encoder_type

    model = RSNABasic(logger=logger,
                      encoder_slug=encoder_slug,
                      pretrained=args.net_pretrained,
                      num_classes=args.net_num_classes,
                      weight_file=weight_file)

    model.init_model()

    return model


class RSNABasic(AbstractModel):
    """Alapvető modellek futtatásához.
    ResNet, Densenet, EfficientNet, Inception, stb.
    """

    def __init__(self, logger=None, encoder_slug: str = 'resnet-18', pretrained=False, num_classes=1,
                 weight_file=None):
        super().__init__(logger, weight_file)

        self.encoder_slug = encoder_slug
        self.num_classes = num_classes
        self.pretrained = pretrained

        self.backbone, num_bottleneck_filters = create_classifier_backbone(encoder_slug=encoder_slug,
                                                                           pretrained=pretrained,
                                                                           num_classes=num_classes)

        # self.dicom_window = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1)
        # self.dicom_activation = nn.Sigmoid()

        # Classification
        # self.feature = nn.Sequential(
        #     nn.Conv2d(in_channels=num_bottleneck_filters, out_channels=128, kernel_size=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 32, kernel_size=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        # )

        # self.logit = nn.Conv2d(32, out_channels=num_classes, kernel_size=1)
        # self.fc = nn.Linear(in_features=num_bottleneck_filters, out_features=num_classes)

        # weight_init(self.dicom_window)
        # weight_init(self.feature)
        # weight_init(self.logit)

    def init_model(self):
        self.load_pretrained_weights()

        if torch.cuda.is_available():

            # self.dicom_window.cuda()
            # self.dicom_activation.cuda()
            self.backbone.cuda()
            # self.feature.cuda()
            # self.logit.cuda()

        self.mode = 'train'

    def forward(self, x):

        # x = self.dicom_window(x)
        # x = self.dicom_activation(x)

        x = self.backbone(x)

        # x = F.dropout(x, 0.5, training=self.training)
        # x = F.adaptive_avg_pool2d(x, 1)
        # x = self.feature(x)
        # x = self.logit(x)

        return x


class RSNAResnet(AbstractModel):

    def __init__(self, logger=None, encoder_slug: str = 'resnet-18', pretrained=False, num_classes=1):
        super().__init__(logger)

        self.encoder_slug = encoder_slug
        self.num_classes = num_classes
        self.pretrained = pretrained

        encoder, num_bottleneck_filters = create_classifier_backbone(encoder_slug=encoder_slug, pretrained=pretrained)

        self.dicom_window = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1)
        self.dicom_activation = nn.Sigmoid()

        # Resnet's original input stem
        self.encoder0 = nn.Sequential(encoder.conv1,
                                      encoder.bn1,
                                      encoder.relu,
                                      encoder.maxpool)

        self.encoder1 = encoder.layer1  # 256
        self.encoder2 = encoder.layer2  # 512
        self.encoder3 = encoder.layer3  # 1024
        self.encoder4 = encoder.layer4  # 2048

        # Classification
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=num_bottleneck_filters, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.logit = nn.Conv2d(32, out_channels=num_classes, kernel_size=1)

        # for m in self.modules():
        #     weight_init(m)

    def init_model(self):
        self.load_pretrained_weights()

        if torch.cuda.is_available():

            self.dicom_window.cuda()
            self.dicom_activation.cuda()
            self.encoder0.cuda()
            self.encoder1.cuda()
            self.encoder2.cuda()
            self.encoder3.cuda()
            self.encoder4.cuda()

            self.feature.cuda()
            self.logit.cuda()

        self.mode = 'train'

    def forward(self, x):

        x = self.dicom_window(x)
        x = self.dicom_activation(x)

        x = self.encoder0(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        x = F.dropout(x, 0.5, training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)

        x = self.feature(x)
        logit = self.logit(x)

        return logit
