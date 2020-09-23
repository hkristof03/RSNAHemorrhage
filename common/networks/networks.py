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
"""
Network package for running ML/DL experiments.

"""
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import Tuple

from common.callbacks import CallbackList, Callback, ExperimentTiming
from common.logger import DefaultLogger
from common.models import AbstractModel


try:
    from apex import amp
except ModuleNotFoundError:
    pass


class AbstractNetwork(ABC):

    def __init__(self, logger: Optional[DefaultLogger] = None, use_amp: bool = False,
                 batch_size: int = 16, accumulation_step: int = 1, iteration_num: int = 100,
                 iteration_valid=100) -> None:
        super().__init__()

        self.__logger: Optional[DefaultLogger] = logger
        self.__optimizer: Optional[Optimizer] = None
        self.__scheduler: Optional[_LRScheduler] = None
        self.__model = None
        self.predictions = None

        self.__callbacks = CallbackList()

        self.__batch_size = batch_size
        self.__iteration_start = 0
        self.__iteration_num = iteration_num
        self.__iteration_valid = iteration_valid
        self.__dataloaders = {}
        self.__use_amp = use_amp
        self.__accumulation_step = accumulation_step

        self._info("")
        self._info("============== NETWORK ==============")
        self._info("    use_amp = {}".format(self.__use_amp))
        self._info("    iteration_num = {}".format(self.__iteration_num))
        self._info("    batch_size = {}".format(self.__batch_size))
        self._info("    accumulation_step = {}".format(self.__accumulation_step))
        self._info("    iteration_valid = {}".format(self.__iteration_valid))

        # Adding general callbacks
        self.add_callback(ExperimentTiming(self.__logger))

        # Init finished.
        self.on_after_init()

    def _info(self, msg):
        """Log 'msg' with severity 'INFO'."""
        if self.__logger is not None:
            self.__logger.info(msg)

    @property
    def optimizer(self) -> Optimizer:
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        self.__optimizer = optimizer

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    @property
    def scheduler(self) -> _LRScheduler:
        return self.__scheduler

    @scheduler.setter
    def scheduler(self, scheduler: _LRScheduler):
        self.__scheduler = scheduler

    @property
    def callbacks(self) -> Union[CallbackList]:
        """CallbackList: list of registered callbacks."""
        return self.__callbacks

    @callbacks.setter
    def callbacks(self, callbacks: CallbackList):
        self.__callbacks = callbacks

    @property
    def iteration_num(self) -> int:
        return self.__iteration_num

    def load_checkpoint(self, checkpoint_file):
        # TODO:
        pass

    def add_callback(self, callback: Callback):
        self.callbacks.add_callbacks(callback)

    def add_dataloader(self, name: str, dataloader: DataLoader):
        self.__dataloaders[name] = dataloader

    def get_dataloader(self, name):
        if name in self.__dataloaders:
            return self.__dataloaders[name]

        raise ValueError("Unknown datalaoder `{}`! You should add your dataloaders to the network. "
                         "See #add_dataloader() method".format(name))

    def on_after_init(self, *args, **kwargs):
        self.callbacks.on_after_init(network=self, *args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        self.callbacks.on_train_begin(network=self, *args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        self.callbacks.on_train_end(network=self, *args, **kwargs)

    def on_epoch_begin(self, epoch: int, *args, **kwargs):
        self.callbacks.on_epoch_begin(epoch, network=self, *args, **kwargs)

    def on_epoch_end(self, epoch: int, *args, **kwargs):
        self.callbacks.on_epoch_end(epoch, network=self, *args, **kwargs)

    def on_iteration_begin(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, *args, **kwargs):
        self.callbacks.on_iteration_begin(epoch, iteration, batch_idx, batch_data, network=self, *args, **kwargs)

    def on_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any,
                         outputs: Tensor, targets: Tensor, loss: Tensor, *args, **kwargs):
        self.callbacks.on_iteration_end(epoch, iteration, batch_idx, batch_data, outputs, targets,
                                        loss=loss, network=self, *args, **kwargs)

    def on_validation_begin(self, epoch, iteration, *args, **kwargs):
        self.callbacks.on_validation_begin(epoch, iteration, *args, **kwargs)

    def on_validation_end(self, epoch, iteration, *args, **kwargs):
        self.callbacks.on_validation_end(epoch, iteration, network=self, *args, *kwargs)

    def on_validation_iteration_begin(self, epoch: int, iteration: int, batch_idx: int, batch_data: any,
                                      *args, **kwargs):
        self.callbacks.on_validation_iteration_begin(epoch, iteration, batch_idx, batch_data, *args, *kwargs)

    def on_validation_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any,
                                    outputs: Tensor, targets: Tensor, loss: Tensor, *args, **kwargs):
        self.callbacks.on_validation_iteration_end(epoch, iteration, batch_idx, batch_data, outputs, targets,
                                                   loss=loss, network=self, *args, **kwargs)

    def on_predict_begin(self, loader: DataLoader, *args, **kwargs):
        pass

    def on_predict_iteration_begin(self, batch_idx: int, batch_data: any):
        pass

    def on_predict_iteration_end(self, batch_idx: int, batch_data: any, outputs: Tensor):
        pass

    def on_predict_end(self, *args, **kwargs):
        pass

    def fit(self):
        self.on_train_begin()

        assert isinstance(self.model, AbstractModel), "The model should be an instance of `common.models.AbstractModel`"
        assert isinstance(self.optimizer, Optimizer), "The optimizer should be an instance of `torch.optim.Optimizer`"

        iteration = self.__iteration_start
        loader = self.get_dataloader('train')
        it = 0
        epoch = 0

        while iteration < self.__iteration_num:

            self.optimizer.zero_grad()

            # Starting a new epoch
            self.on_epoch_begin(epoch)

            for batch_idx, batch_data in enumerate(loader):

                iteration = it + self.__iteration_start

                if iteration > self.__iteration_num:
                    # break train loader loop
                    break

                # Iteration starts
                self.on_iteration_begin(epoch, iteration, batch_idx=batch_idx, batch_data=batch_data)

                # Model's output (usually logits)
                outputs, targets = self._fit_batch(epoch, iteration, batch_idx=batch_idx, batch_data=batch_data)

                # Calculate loss
                loss = self._compute_loss(outputs=outputs, target=targets)

                if self.__use_amp:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # accumulation_step
                if batch_idx % self.__accumulation_step == 0:
                    if not self.__use_amp:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    else:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 0.5)

                    self.optimizer.step(None)
                    self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                # Iteration end
                self.on_iteration_end(epoch, iteration,
                                      batch_idx=batch_idx,
                                      batch_data=batch_data,
                                      outputs=outputs,
                                      targets=targets,
                                      loss=loss)

                del outputs, targets, loss

                self.validate(epoch, iteration)

                # Iteration counter
                it = it + 1

                if self.callbacks.training_break():
                    break

            # End of an epoch
            self.on_epoch_end(epoch)

            # Epoch counter
            epoch = epoch + 1

            if self.callbacks.training_break():
                break

        self.on_train_end()

    @abstractmethod
    def _fit_batch(self, epoch: int, iteration: int, batch_idx: int,
                   batch_data: Tuple[Tensor, Tensor, Tensor, any]) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _compute_loss(self, outputs, target):
        raise NotImplementedError

    def validate(self, epoch, iteration):

        # is validation iter?
        if (iteration + 1) % self.__iteration_valid == 0:

            with torch.no_grad():

                self.model.mode = 'eval'
                self.on_validation_begin(epoch, iteration)

                loader = self.get_dataloader('validation')

                for batch_idx, batch_data in enumerate(loader):
                    self.on_validation_iteration_begin(epoch, iteration, batch_idx, batch_data)

                    outputs, targets = self._fit_validation_batch(epoch, iteration,
                                                                  batch_idx=batch_idx,
                                                                  batch_data=batch_data)

                    # Calculate loss
                    loss = self._compute_loss(outputs=outputs, target=targets)

                    self.on_validation_iteration_end(epoch, iteration,
                                                     batch_idx=batch_idx,
                                                     batch_data=batch_data,
                                                     outputs=outputs,
                                                     targets=targets,
                                                     loss=loss)

                self.on_validation_end(epoch, iteration)

                del outputs, targets, loss
                self.model.mode = 'train'

    def _fit_validation_batch(self, epoch: int, iteration: int, batch_idx: int,
                              batch_data: Tuple[Tensor, Tensor, Tensor, any]) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def predict(self):

        self.model.mode = 'eval'

        with torch.no_grad():

            loader = self.get_dataloader('test')

            self.on_predict_begin(loader)

            for batch_idx, batch_data in enumerate(loader):

                self.on_predict_iteration_begin(batch_idx, batch_data)

                outputs = self._predict_batch(batch_idx=batch_idx, batch_data=batch_data)

                self.on_predict_iteration_end(batch_idx, batch_data, outputs=outputs)

            self.on_predict_end()

        return self.predictions

    def _predict_batch(self, batch_idx: int, batch_data: Tuple[Tensor, Tensor, Tensor, any]) -> Tensor:
        raise NotImplementedError
