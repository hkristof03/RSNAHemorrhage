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
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from common.logger import DefaultLogger
from common.networks import ImageClassifierNetwork
from typing import Optional


class RSNANetwork(ImageClassifierNetwork):

    def __init__(self, logger: Optional[DefaultLogger] = None, use_amp: bool = False, batch_size: int = 16,
                 accumulation_step: int = 1, iteration_num: int = 100, iteration_valid=100) -> None:
        super().__init__(logger, use_amp, batch_size, accumulation_step, iteration_num, iteration_valid)

        self.classes = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
        self.__any_weight = 2.6

    def on_iteration_begin(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, *args, **kwargs):
        super().on_iteration_begin(epoch, iteration, batch_idx, batch_data, *args, **kwargs)

        progress = iteration / self.iteration_num
        # self._info("Progress: {}".format(progress))

        if progress < 0.2:
            self.__any_weight = 2.6
        elif progress < 0.4:
            self.__any_weight = 2.4
        elif progress < 0.6:
            self.__any_weight = 2.2
        elif progress < 0.8:
            self.__any_weight = 2.0
        else:
            self.__any_weight = 1.0

    def on_validation_begin(self, epoch, iteration, *args, **kwargs):
        super().on_validation_begin(epoch, iteration, *args, **kwargs)

        # Reset to default during validation
        self.__any_weight = 2.0

    def _compute_loss(self, outputs, target):

        # Normal sampler
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, self.__any_weight]).float().cuda()

        loss = F.binary_cross_entropy_with_logits(outputs, target, weight=weights)

        return loss

    def on_predict_begin(self, loader: DataLoader, *args, **kwargs):
        super().on_predict_begin(loader, *args, **kwargs)
        self.predictions = []

    def on_predict_iteration_end(self, batch_idx: int, batch_data: any, outputs: Tensor):
        super().on_predict_iteration_end(batch_idx, batch_data, outputs)

        outputs = outputs.squeeze(-1).squeeze(-1)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.data.cpu().numpy()

        # outputs = (outputs > 0.5).astype(np.uint8)

        indices, images, _, extras = batch_data

        if batch_idx % 250 == 0:
            self._info("Predicting batch #{}".format(batch_idx))

        for i, row in enumerate(extras):
            for cls_i, cls in enumerate(self.classes):
                self.predictions.append({'ID': '{}_{}'.format(row['id'], cls), 'Label': outputs[i][cls_i]})

    def on_predict_end(self, *args, **kwargs):
        self.predictions = pd.DataFrame(self.predictions)


class RSNAAnyClassifierNetwork(RSNANetwork):

    def __init__(self, logger: Optional[DefaultLogger] = None, use_amp: bool = False, batch_size: int = 16,
                 accumulation_step: int = 1, iteration_num: int = 100, iteration_valid=100) -> None:
        super().__init__(logger, use_amp, batch_size, accumulation_step, iteration_num, iteration_valid)

        self.classes = ['any']

    def _compute_loss(self, outputs, target):
        batch_size, num_class, H, W = outputs.shape

        logit = outputs.view(batch_size, num_class)
        truth = target.view(batch_size, num_class)

        assert (logit.shape == truth.shape)

        # Normal sampler
        # weights = torch.tensor([6]).float().cuda()
        weights = None

        loss = F.binary_cross_entropy_with_logits(logit, truth, pos_weight=weights)

        return loss

    def on_predict_iteration_end(self, batch_idx: int, batch_data: any, outputs: Tensor):

        outputs = outputs.squeeze(-1).squeeze(-1)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.data.cpu().numpy()

        # outputs = (outputs > 0.5).astype(np.uint8)

        indices, images, _, extras = batch_data

        if batch_idx % 250 == 0:
            self._info("Predicting batch #{}".format(batch_idx))

        for i, row in enumerate(extras):
            for cls_i, cls in enumerate(self.classes):
                self.predictions.append({'ID': '{}_{}'.format(row['id'], cls), 'Any': outputs[i][cls_i]})
