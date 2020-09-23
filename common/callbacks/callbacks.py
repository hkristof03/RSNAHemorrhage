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
from abc import ABC
from torch import Tensor
from common.logger import DefaultLogger
from typing import Optional


class CallbackList:

    def __init__(self, callbacks=None):
        if callbacks is None:
            self.callbacks = []
        elif isinstance(callbacks, Callback):
            self.callbacks = [callbacks]
        else:
            self.callbacks = callbacks

    def __len__(self):
        return len(self.callbacks)

    def add_callbacks(self, callbacks):
        if self.callbacks is None:
            self.callbacks = []

        if isinstance(callbacks, Callback):
            self.callbacks.append(callbacks)
        else:
            for callback in callbacks:
                self.callbacks.append(callback)

    def set_params(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.set_params(*args, **kwargs)

    def on_after_init(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_after_init(*args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(*args, **kwargs)

    def on_epoch_begin(self, epoch: int, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, *args, **kwargs)

    def on_epoch_end(self, epoch: int, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, *args, **kwargs)

    def on_iteration_begin(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_iteration_begin(epoch, iteration, batch_idx, batch_data, *args, **kwargs)

    def on_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any,
                         outputs: Tensor, targets: Tensor, loss: Tensor, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_iteration_end(epoch, iteration, batch_idx, batch_data, outputs, targets, loss, *args, **kwargs)

    def on_validation_begin(self, epoch, iteration, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_validation_begin(epoch, iteration, *args, **kwargs)

    def on_validation_end(self, epoch, iteration, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_validation_end(epoch, iteration, *args, **kwargs)

    def on_validation_iteration_begin(self, epoch: int, iteration: int, batch_idx: int, batch_data: any,
                                      *args, **kwargs):
        for callback in self.callbacks:
            callback.on_validation_iteration_begin(epoch, iteration, batch_idx, batch_data, *args, **kwargs)

    def on_validation_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any,
                                    outputs: Tensor, targets: Tensor, loss: Tensor, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_validation_iteration_end(epoch, iteration, batch_idx, batch_data, outputs, targets,
                                                 loss, *args, **kwargs)

    def training_break(self, *args, **kwargs):
        callback_out = [callback.training_break(*args, **kwargs) for callback in self.callbacks]
        return any(callback_out)


class Callback(ABC):

    def __init__(self, logger: Optional[DefaultLogger] = None):
        self.__logger = logger

    def _info(self, msg):
        """Log 'msg' with severity 'INFO'."""
        if self.__logger is not None:
            self.__logger.info(msg)

    def send_metric(self, channel_name, x, y=None, timestamp=None):
        self.__logger.send_metric(channel_name, x, y, timestamp)

    def on_after_init(self, *args, **kwargs):
        pass

    def on_train_begin(self, *args, **kwargs):
        pass

    def on_train_end(self, *args, **kwargs):
        pass

    def on_epoch_begin(self, epoch: int, *args, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, *args, **kwargs):
        pass

    def on_iteration_begin(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, *args, **kwargs):
        pass

    def on_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any,
                         outputs: Tensor, targets: Tensor, loss: Tensor, *args, **kwargs):
        pass

    def on_validation_begin(self, epoch, iteration, *args, **kwargs):
        pass

    def on_validation_end(self, epoch, iteration, *args, **kwargs):
        pass

    def on_validation_iteration_begin(self, epoch: int, iteration: int, batch_idx: int, batch_data: any,
                                      *args, **kwargs):
        pass

    def on_validation_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any,
                                    outputs: Tensor, targets: Tensor, loss: Tensor, *args, **kwargs):
        pass

    def training_break(self, *args, **kwargs):
        return False

