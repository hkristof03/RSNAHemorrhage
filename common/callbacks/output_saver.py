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
from torch import Tensor

from common.callbacks import Callback
from common.datasets import AbstractDataset
from common.helpers import Averager, persist_torch_model
from common.logger import DefaultLogger
from common.networks import AbstractNetwork

from torch.utils.data import DataLoader
from typing import Optional


class OutputSaver(Callback):

    def __init__(self, logger: Optional[DefaultLogger] = None, best_in_iter: int = 5000,
                 output_dir: str = './'):
        super().__init__(logger)

        self._averager = Averager()
        self.__best_in_iter = best_in_iter
        self.__output_dir = output_dir
        self.__best = np.inf
        self.__stage = -1
        self.__predicitons = []

    def on_validation_begin(self, epoch, iteration, *args, **kwargs):

        if iteration % self.__best_in_iter == 0:
            self._averager.reset()
            self.__best = np.inf
            self.__stage = self.__stage + 1
            self.__predicitons = []

    def on_validation_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor,
                                    targets: Tensor, loss: Tensor, *args, **kwargs):
        super().on_validation_iteration_end(epoch, iteration, batch_idx, batch_data, outputs, targets, loss, *args,
                                            **kwargs)
        indices, images, targets, extras = batch_data

        self._averager.send(loss.item())

        outputs = torch.sigmoid(outputs)

        np_targets = targets.view(-1).data.cpu().numpy()
        np_outputs = outputs.view(-1).data.cpu().numpy()

        for i, index in enumerate(indices):
            self.__predicitons.append({
                'id': extras[i]['id'],
                'label': np_targets[i],
                'prediction': np_outputs[i]
            })

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

        file = "{}/outputs__ds_{}__fold_{}__best_in_stage_{}.csv".format(
            self.__output_dir, dataset, datafold, self.__stage)

        self._info("Saving best prediction (based on best valid loss) result to {}...".format(file))
        df = pd.DataFrame(self.__predicitons)
        df.to_csv(file, index=False)

        self.__predicitons = []
