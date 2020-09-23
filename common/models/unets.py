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

from . import AbstractModel, create_classifier_backbone, weight_init


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),

            nn.Conv2d(out_channel // 2, out_channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),

            nn.Conv2d(out_channel // 2, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        for m in self.top.modules():
            weight_init(m)

    def forward(self, x):
        x = self.top(torch.cat(x, 1))
        return x


class UnetResnet(AbstractModel):

    def __init__(self, logger=None, encoder_slug: str = 'resnet-18', pretrained=False, num_classes=1):
        super().__init__(logger)

        self.encoder_slug = encoder_slug
        self.num_classes = num_classes
        self.pretrained = pretrained

        encoder, num_bottleneck_filters = create_classifier_backbone(encoder_slug=encoder_slug, pretrained=pretrained)

        # Resnet's original input stem
        self.encoder0 = nn.Sequential(encoder.conv1,
                                      encoder.bn1,
                                      encoder.relu,
                                      encoder.maxpool)

        self.encoder1 = encoder.layer1  # 256
        self.encoder2 = encoder.layer2  # 512
        self.encoder3 = encoder.layer3  # 1024
        self.encoder4 = encoder.layer4  # 2048

        self.decoder4 = Decoder(num_bottleneck_filters, 128)
        self.decoder3 = Decoder(num_bottleneck_filters // 2 + 128, 128)
        self.decoder2 = Decoder(num_bottleneck_filters // 4 + 128, 128)
        self.decoder1 = Decoder(num_bottleneck_filters // 8 + 128, 128)
        self.decoder0 = Decoder(num_bottleneck_filters // 8 + 128, 128)

        self.logit = nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Conv2d(128, num_classes, kernel_size=1, bias=True)
        )

        # For focal loss initialization
        # pi = 0.01
        # self.logit.weight.data.fill_(0)
        # self.logit.bias.data.fill_(-np.log((1 - pi) / pi))

        for m in self.logit.modules():
            weight_init(m)

    # noinspection PyUnresolvedReferences,PyArgumentList
    def init_model(self):
        self.load_pretrained_weights()

        if torch.cuda.is_available():
            self.encoder0.cuda()
            self.encoder1.cuda()
            self.encoder2.cuda()
            self.encoder3.cuda()
            self.encoder4.cuda()

            self.decoder0.cuda()
            self.decoder1.cuda()
            self.decoder2.cuda()
            self.decoder3.cuda()
            self.decoder4.cuda()

            self.logit.cuda()

        self.set_mode('train')

    def metric(self, outputs, labels):
        pass

    def forward(self, x):
        batch_size, C, H, W = x.shape

        # x: B x 3 x 1/1 x 1/1
        e0 = self.encoder0(x)  # B x 64 x 1/4 x 1/4

        # ######################
        #  ENCODER BLOCKS
        # ######################
        e1 = self.encoder1(e0)  # B x 256 x 1/4 x 1/4
        e2 = self.encoder2(e1)  # B x 512 x 1/8 x 1/8
        e3 = self.encoder3(e2)  # B x 1024 x 1/16 x 1/16
        e4 = self.encoder4(e3)  # B x 2048 x 1/32 x 1/32

        # ######################
        #  DECODER BLOCKS
        # ######################
        d4 = self.decoder4([e4, ])  # B x 128 x 1/32 x 1/32
        d3 = self.decoder3([e3, F.interpolate(d4, scale_factor=2, mode='nearest')])  # B x 128 x 1/16 x 1/16
        d2 = self.decoder2([e2, F.interpolate(d3, scale_factor=2, mode='nearest')])  # B x 128 x 1/8 x 1/8
        d1 = self.decoder1([e1, F.interpolate(d2, scale_factor=2, mode='nearest')])  # B x 128 x 1/4 x 1/4
        d0 = self.decoder0([e0, d1])  # B x 128 x 1/4 x 1/4

        logit = self.logit(d0)
        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)

        return logit
