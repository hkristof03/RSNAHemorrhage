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
from datetime import datetime

from torch import Tensor

from common.callbacks import Callback


class ExperimentTiming(Callback):

    def __init__(self, logger=None):
        super().__init__(logger)

        self.__epoch_started = None
        self.__iter_started = None
        self.__iter_elapsed = 0

    @staticmethod
    def factory(logger=None, *args, **kwargs):
        return ExperimentTiming(logger=logger)

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self._info("Starting epoch {}".format(epoch))
        self.__epoch_started = datetime.now()

    def on_iteration_begin(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, *args, **kwargs):

        if self.__iter_started is None:
            self.__iter_started = datetime.now()

        self.__iter_elapsed += 1

    def on_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor,
                         targets: Tensor, loss: Tensor, *args, **kwargs):
        super().on_iteration_end(epoch, iteration, batch_idx, batch_data, outputs, targets, loss, *args, **kwargs)

        if self.__iter_elapsed % 1000 == 0:

            iter_time = datetime.now() - self.__iter_started
            time = str(iter_time)[:-7]

            self._info("Last 1000 iteration finished in: {}".format(time))

            self.__iter_started = None
            self.__iter_elapsed = 0


    def on_epoch_end(self, epoch, *args, **kwargs):
        super().on_epoch_end(epoch, *args, **kwargs)

        epoch_time = datetime.now() - self.__epoch_started
        time = str(epoch_time)[:-7]

        self._info("Epoch {0} finished in {1}".format(epoch, time))
