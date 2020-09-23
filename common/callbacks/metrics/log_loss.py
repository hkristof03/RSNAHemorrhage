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
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from sklearn.metrics import log_loss

from common.callbacks.metrics import AbstractMetric
from common.logger import DefaultLogger
from common.helpers import Averager

EPS = 1e-15


class MultilabelLogLoss(AbstractMetric):

    def __init__(self, logger: Optional[DefaultLogger] = None, save_best: bool = False, output_dir: str = './',
                 class_names=None, log_iteration: int = 100):
        super().__init__(logger, save_best, minimize=True, output_dir=output_dir)

        # Optional class names for logging (`cls_i` use otherwise)
        self.__class_names = class_names

        assert self.__class_names is not None

        # Calculate and send train logs after `log_iteration` iterations.
        self.__log_iteration = log_iteration

        # Averagers per class
        self.__averagers = {}

    def on_train_begin(self, *args, **kwargs):

        if self.__log_iteration <= 0:
            return

        for log_name in self.__class_names:
            self.__averagers['train_' + log_name] = Averager()
            self.__averagers['valid_' + log_name] = Averager()

    def on_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor,
                         targets: Tensor, loss: Tensor, *args, **kwargs):

        if self.__log_iteration <= 0:
            return

        # add iteration train log
        for i in range(len(self.__class_names)):
            try:
                self.__averagers['train_' + self.__class_names[i]].send(log_loss(
                    targets.data.cpu().numpy()[:, i],
                    np.clip(F.sigmoid(outputs).data.cpu().numpy()[:, i], EPS, 1 - EPS)
                ))
            except ValueError:
                self.__averagers['train_' + self.__class_names[i]].send(1)

        # check if % iter == 0
        if (iteration % self.__log_iteration == 0) and (iteration > 0):
            # write/send logs
            self._info("")
            self._info("----- {} LOG #{} -----".format('Train', iteration))

            for i in range(len(self.__class_names)):
                avg = self.__averagers['train_' + self.__class_names[i]]
                self._info("  {}: {}".format(self.__class_names[i], avg.value))
                self.send_metric('Train - ' + self.__class_names[i] + ' - LogLoss', x=iteration, y=avg.value)
                avg.reset()

    def on_validation_begin(self, epoch, iteration, *args, **kwargs):
        # Global (weighted) averager
        self._averager.reset()

        for log_name in self.__class_names:
            self.__averagers['valid_' + log_name].reset()

    def on_validation_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor,
                                    targets: Tensor, loss: Tensor, *args, **kwargs):

        # Calculate global (weighted) log-loss
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0]).float().cuda()
        lloss = F.binary_cross_entropy_with_logits(outputs, targets, weight=weights)

        self._averager.send(lloss)

        # Calculate per-class log-loss
        for i in range(len(self.__class_names)):
            try:
                self.__averagers['valid_' + self.__class_names[i]].send(log_loss(
                    targets.data.cpu().numpy()[:, i],
                    np.clip(F.sigmoid(outputs).data.cpu().numpy()[:, i], EPS, 1 - EPS)
                ))
            except ValueError:
                self.__averagers['valid_' + self.__class_names[i]].send(1)

    def on_validation_end(self, epoch, iteration, *args, **kwargs):
        super().on_validation_end(epoch, iteration, *args, **kwargs)

        lloss = self._averager.value

        self._info("Validation log-loss {}".format(lloss))
        self.send_metric('VALIDATION - Weighted Log-loss', x=iteration, y=lloss)

        # write/send logs
        self._info("")
        self._info("----- {} LOG #{} -----".format('Valid', iteration))

        for i in range(len(self.__class_names)):
            avg = self.__averagers['valid_' + self.__class_names[i]]
            self._info("  {}: {}".format(self.__class_names[i], avg.value))
            self.send_metric('Valid - ' + self.__class_names[i] + ' - LogLoss', x=iteration, y=avg.value)
            avg.reset()

    def _get_metric_name(self) -> str:
        return 'log_loss'
