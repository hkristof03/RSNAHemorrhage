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

from common.callbacks import Callback
from common.datasets import AbstractDataset
from common.helpers import Averager, persist_torch_model
from common.logger import DefaultLogger
from common.networks import AbstractNetwork

from torch.utils.data import DataLoader
from typing import Optional


class ModelCheckpoint(Callback):

    def __init__(self, logger: Optional[DefaultLogger] = None, best_in_iter: int = 5000,
                 output_dir: str = './'):
        super().__init__(logger)

        self._averager = Averager()
        self.__best_in_iter = best_in_iter
        self.__output_dir = output_dir
        self.__best = np.inf
        self.__stage = -1

        if self.__best_in_iter == 0:
            self.__best_in_iter = 5000

    def on_validation_begin(self, epoch, iteration, *args, **kwargs):

        if iteration % self.__best_in_iter == 0:
            self._averager.reset()
            self.__best = np.inf
            self.__stage = self.__stage + 1

    def on_validation_end(self, epoch, iteration, *args, **kwargs):
        super().on_validation_end(epoch, iteration, *args, **kwargs)

        network = kwargs['network'] if 'network' in kwargs else None  # type: Optional[AbstractNetwork]

        if network is None:
            return

        lloss = self._averager.value

        if lloss < self.__best:
            self.__best = lloss
            self.__save_best_validation_result(epoch, iteration, network)

    def __save_best_validation_result(self, epoch, iteration, network: AbstractNetwork):
        dl = network.get_dataloader('validation')  # type: DataLoader
        # noinspection PyTypeChecker
        ds = dl.dataset  # type: AbstractDataset

        dataset = ds.data_set
        datafold = ds.data_fold

        file = "{}/ds_{}__fold_{}__best_in_stage_{}.pt".format(self.__output_dir, dataset, datafold, self.__stage)

        self._info("Saving best iteration result to {}...".format(file))
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
