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

from typing import Optional, Union, List
from torch import Tensor
from sklearn.metrics import precision_recall_fscore_support

from common.callbacks import Callback
from common.logger import DefaultLogger


class Averager:
    def __init__(self):
        self.current_total = None
        self.iterations = None

    def send(self, value, iterations=0):

        if self.current_total is None:
            self.current_total = value
            self.iterations = iterations

        self.current_total += value
        self.iterations += iterations

    def value(self):
        if self.iterations is None:
            return 0
        else:
            if self.iterations == 0:
                self.iterations = 1

            return 1.0 * self.current_total / self.iterations

    def reset(self, totals=None, iterations=None):
        self.current_total = totals
        self.iterations = iterations


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    # true_positives = torch.sum(confusion_vector == 1).item()
    # false_positives = torch.sum(confusion_vector == float('inf')).item()
    # true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    # false_negatives = torch.sum(confusion_vector == 0).item()

    true_positives = confusion_vector == 1
    false_positives = confusion_vector == float('inf')
    true_negatives = torch.isnan(confusion_vector)
    false_negatives = confusion_vector == 0

    return true_positives, false_positives, true_negatives, false_negatives


class FScoreMultilabel(Callback):

    def __init__(self, logger: Optional[DefaultLogger] = None, log_iteration: int = 100,
                 log_validation_average=False, class_names=None,
                 thresholds: Optional[Union[int, List]] = None):
        super().__init__(logger)

        # Calculate and send train logs after `log_iteration` iterations.
        self.__log_iteration = log_iteration

        # Calculate and send logs on end of validation
        self.__log_validation_average = log_validation_average

        # Optional class names for logging (`cls_i` use otherwise)
        self.__class_names = class_names

        # Thresholds for sigmoid -> prediciton (list: per labels; scalar: for all labels)
        self.__thresholds = thresholds

        if self.__thresholds is None:
            self.__thresholds = 0.5

        self.__averagers = {}

    def on_train_begin(self, *args, **kwargs):

        if self.__log_iteration <= 0:
            return

        logs = ['train_tp', 'train_fp', 'train_tn', 'train_fn']

        for log_name in logs:
            self.__averagers[log_name] = Averager()

    def on_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor,
                         targets: Tensor, loss: Tensor, *args, **kwargs):

        if self.__log_iteration <= 0:
            return

        # add iteration train log
        self.__calculate_logs('train', outputs, targets)

        # check if % iter == 0
        if (iteration % self.__log_iteration == 0) and (iteration > 0):
            # write/send logs
            self.__write_logs('train', 'TRAIN', iteration)

            # reset train averagers
            logs = ['train_tp', 'train_fp', 'train_tn', 'train_fn']

            for log_name in logs:
                self.__averagers[log_name].reset()

    def on_validation_begin(self, epoch, iteration, *args, **kwargs):

        if not self.__log_validation_average:
            return

        logs = ['val_tp', 'val_fp', 'val_tn', 'val_fn']

        for log_name in logs:
            if log_name not in self.__averagers:
                self.__averagers[log_name] = Averager()

            # Reset validation log at every cycle.
            self.__averagers[log_name].reset()

    def on_validation_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor,
                                    targets: Tensor, loss: Tensor, *args, **kwargs):

        if not self.__log_validation_average:
            return

        self.__calculate_logs('val', outputs, targets)

    def on_validation_end(self, epoch, iteration, *args, **kwargs):

        if not self.__log_validation_average:
            return

        self.__write_logs('val', 'VALID', iteration)

    def __get_class_name(self, i, prefix='Train'):
        if self.__class_names is not None and len(self.__class_names) > i:
            return prefix + ' - ' + self.__class_names[i]

        return '{} - Class #{}'.format(prefix, i)

    def __calculate_logs(self, prefix: str, outputs: Tensor, targets: Tensor):
        outputs_sigmoid = F.sigmoid(outputs)
        outputs_sigmoid = (outputs_sigmoid > self.__thresholds).float()
        outputs_sigmoid = outputs_sigmoid.squeeze(-1).squeeze(-1)

        targets = (targets > 0.5).float()

        tp, fp, tn, fn = confusion(outputs_sigmoid, targets)

        self.__averagers['{}_tp'.format(prefix)].send(tp.sum(0).data.cpu().numpy())
        self.__averagers['{}_fp'.format(prefix)].send(fp.sum(0).data.cpu().numpy())
        self.__averagers['{}_tn'.format(prefix)].send(tn.sum(0).data.cpu().numpy())
        self.__averagers['{}_fn'.format(prefix)].send(fn.sum(0).data.cpu().numpy())

    def __write_logs(self, prefix: str, title: str, iteration):

        tp = self.__averagers['{}_tp'.format(prefix)].value()
        fp = self.__averagers['{}_fp'.format(prefix)].value()
        tn = self.__averagers['{}_tn'.format(prefix)].value()
        fn = self.__averagers['{}_fn'.format(prefix)].value()

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
        # all = tp + fp + tn + fn

        self._info("")
        self._info("----- {} LOG #{} -----".format(title, iteration))
        self._info("  TP: {}".format(tp))
        self._info("  FP: {}".format(fp))
        self._info("  TN: {}".format(tn))
        self._info("  FN: {}".format(fn))
        self._info("")
        self._info("  PR: {}".format(precision))
        self._info("  RC: {}".format(recall))
        self._info("  F1: {}".format(f1))

        for i in range(len(tp)):
            channel_name = self.__get_class_name(i, title)

            self.send_metric(channel_name + ' - Precision', x=iteration, y=precision[i])
            self.send_metric(channel_name + ' - Recall', x=iteration, y=recall[i])
            self.send_metric(channel_name + ' - F1', x=iteration, y=f1[i])

        self._info("--------------------------")
