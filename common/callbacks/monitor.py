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
from typing import Optional

from torch import Tensor

from common.callbacks import Callback
from common.helpers import Averager
from common.logger import DefaultLogger


class TrainingMonitor(Callback):

    def __init__(self, logger=None, log_every_iteration=10):
        super().__init__(logger)

        self.__log_every_iteration = log_every_iteration
        self.__loss_averagers = {}

    def on_train_begin(self, *args, **kwargs):

        if 'loss' not in self.__loss_averagers:
            self.__loss_averagers['loss'] = Averager()

    def on_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor,
                         targets: Tensor, loss: Tensor, *args, **kwargs):

        averager = self.__loss_averagers['loss']
        averager.send(loss.item())

        if iteration % self.__log_every_iteration == 0:
            # self._info("Iteration Loss - {0}: {1:.5f}".format(iteration, averager.value))
            self.send_metric(channel_name='Iteration Loss', x=iteration, y=averager.value)

        # Report every batch's loss
        self.send_metric(channel_name='Batch Loss', x=iteration, y=loss.item())

    def on_epoch_begin(self, epoch: int, *args, **kwargs):
        averager = self.__loss_averagers['loss']
        averager.reset()

    def on_epoch_end(self, epoch: int, *args, **kwargs):
        averager = self.__loss_averagers['loss']

        self._info("Epoch Loss - {0}: {1:.5f}".format(epoch, averager.value))
        self.send_metric(channel_name='Epoch Loss', x=epoch, y=averager.value)

    def on_validation_begin(self, epoch, iteration, *args, **kwargs):

        if 'valid_loss' not in self.__loss_averagers:
            self.__loss_averagers['valid_loss'] = Averager()

        self.__loss_averagers['valid_loss'].reset()

    def on_validation_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor,
                                    targets: Tensor, loss: Tensor, *args, **kwargs):

        averager = self.__loss_averagers['valid_loss']
        averager.send(loss.item())

    def on_validation_end(self, epoch, iteration, *args, **kwargs):

        averager = self.__loss_averagers['valid_loss']
        self._info("Validation Loss - {0}: {1:.5f}".format(iteration, averager.value))
        self.send_metric(channel_name='Validation Loss', x=iteration, y=averager.value)


class LearningRateMonitor(Callback):

    def __init__(self, logger: Optional[DefaultLogger] = None, log_every_iteration=10):
        super().__init__(logger)

        self.__log_every_iteration = log_every_iteration

    def on_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor,
                         targets: Tensor, loss: Tensor, *args, **kwargs):
        super().on_iteration_end(epoch, iteration, batch_idx, batch_data, outputs, targets, loss, *args, **kwargs)

        network = kwargs['network'] if 'network' in kwargs else None

        if network is not None:
            lr = network.optimizer.state_dict()['param_groups'][0]['lr']
            self.send_metric(channel_name='Learning Rate', x=iteration, y=lr)
