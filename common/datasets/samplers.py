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
import random

from torch.utils.data.sampler import Sampler
from common.datasets import AbstractDataset

from typing import Tuple


class BalancedSampler(Sampler):

    def __init__(self, data_source: AbstractDataset) -> None:
        super().__init__(data_source)

        self.__indices = []
        self.__label_cols = data_source.get_label_column_name()
        self.__length_min = np.inf

        df = data_source.dataframe

        for i, cls in enumerate(self.__label_cols):
            idx = df[df[cls] == 1].index.values
            self.__indices.append(idx)
            if len(idx) < self.__length_min:
                self.__length_min = len(idx)

        self.__length = self.__length_min * len(self.__label_cols)

    def __iter__(self):
        indices = []

        for i, cls in enumerate(self.__label_cols):
            indices.append(np.random.choice(self.__indices[i], self.__length_min, replace=False))

        indices = np.stack(indices).T
        indices = indices.reshape(-1)

        random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        return self.__length


class PositiveClassRatioSampler(Sampler):

    def __init__(self, data_source: AbstractDataset, num_iter: int = 10000, batch_size: int = 32,
                 pos_range: Tuple[float, float] = (0.5, 0.5), update_at_every_iter: int = 1000) -> None:
        super().__init__(data_source)

        self.__indices = []
        self.__label_cols = data_source.get_label_column_name()
        self.__batch_size = batch_size
        self.__num_iter = num_iter
        self.__update_at_every_iter = update_at_every_iter
        self.__pos_range = pos_range

        self.__neg_indices = data_source.get_negative_sample_indices()
        self.__pos_indices = data_source.get_positive_sample_indices()

    def __iter__(self):

        indices = []

        num_of_updates = self.__num_iter // self.__update_at_every_iter
        num_of_sample_in_updates = self.__num_iter * self.__batch_size // num_of_updates

        print("   PositiveClassRatioSampler")
        print("   ----------------------------------")
        print("   number of positive sample ratio update during training: {}".format(num_of_updates))
        print("   number of sample in one update iteration: {}".format(num_of_sample_in_updates))

        for i in range(num_of_updates):
            num_of_pos = self.__pos_range[0] - i * (self.__pos_range[0] - self.__pos_range[1]) / num_of_updates
            pos = int(num_of_sample_in_updates * num_of_pos)
            neg = num_of_sample_in_updates - pos

            pos = np.random.choice(self.__pos_indices, pos, replace=False)
            neg = np.random.choice(self.__neg_indices, neg, replace=False)

            updates = np.concatenate((pos, neg))
            np.random.shuffle(updates)

            indices.append(updates)

        indices = np.stack(indices)
        indices = indices.reshape(-1)

        return iter(indices)

    def __len__(self):
        return self.__batch_size * self.__num_iter
