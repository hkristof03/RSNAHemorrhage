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

from abc import ABC, abstractmethod
from common.callbacks import Callback
from common.datasets import AbstractDataset
from common.logger import DefaultLogger
from common.helpers import Averager, persist_torch_model
from common.networks import AbstractNetwork

from torch.utils.data import DataLoader
from typing import Optional


class AbstractMetric(Callback, ABC):

    def __init__(self, logger: Optional[DefaultLogger] = None, save_best: bool = False, minimize: bool = False,
                 output_dir: str = './'):
        super().__init__(logger)

        self._averager = Averager()
        self.__save_best = save_best
        self.__output_dir = output_dir
        self.__minimize = False
        self.__best = np.inf if minimize else -np.inf

    def on_validation_end(self, epoch, iteration, *args, **kwargs):
        super().on_validation_end(epoch, iteration, *args, **kwargs)

        network = kwargs['network'] if 'network' in kwargs else None  # type: Optional[AbstractNetwork]

        if network is None:
            return

        lloss = self._averager.value

        if (self.__minimize and lloss > self.__best) or (not self.__minimize and lloss < self.__best):
            self.__best = lloss
            self.__save_best_validation_result(epoch, iteration, network)

    @abstractmethod
    def _get_metric_name(self) -> str:
        raise NotImplementedError

    def __save_best_validation_result(self, epoch, iteration, network: AbstractNetwork):
        dl = network.get_dataloader('validation')  # type: DataLoader
        # noinspection PyTypeChecker
        ds = dl.dataset  # type: AbstractDataset

        dataset = ds.data_set
        datafold = ds.data_fold

        file = "{}/ds_{}__fold_{}__best_{}.pt".format(self.__output_dir, dataset, datafold, self._get_metric_name())

        self._info("Saving best `{}` result to {}...".format(self._get_metric_name(), file))
        model_mode = network.model.mode
        network.model.mode = 'eval'

        if torch.cuda.is_available():
            network.model.cpu()

            model_state_dict = network.model.state_dict()
            optimizer_state_dict = network.optimizer.state_dict()

            network.model.cuda()
        else:
            model_state_dict = network.model.state_dict()
            optimizer_state_dict = network.optimizer.state_dict()

        persist_torch_model(file, epoch, iteration, model_state_dict, optimizer_state_dict)

        network.model.mode = model_mode
