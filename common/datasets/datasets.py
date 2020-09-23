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

from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from typing import Optional, Callable

from common.logger import DefaultLogger


class AbstractDataset(Dataset, ABC):

    def __init__(self, df: pd.DataFrame, df_type: str, logger: Optional[DefaultLogger] = None,
                 label_smoothing: Optional[float] = None, num_classes: int = 2) -> None:
        super().__init__()

        self.__logger: Optional[DefaultLogger] = logger
        self.__df_type = df_type
        self.__df: pd.DataFrame = df
        self.__collate_fn = None
        self.__data_set = None
        self.__data_fold = None
        self.__label_smoothing = label_smoothing
        self.__num_classes = num_classes

        self._info("")
        self._info("============== DATASET ==============")
        self._info("    df_type = {}".format(self.__df_type))
        self._info("    df shape = {}".format(self.__df.shape))

    def _info(self, msg):
        """Log 'msg' with severity 'INFO'."""
        if self.__logger is not None:
            self.__logger.info(msg)

    def get_data_loader(self, batch_size=1, num_worker=-1, pin_memory=False, drop_last=False, sampler=None):

        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_worker,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=self._get_sampler(dataset=self) if sampler is None else sampler,
            collate_fn=self.collate_fn
        )

        return dataloader

    @property
    def data_set(self):
        return self.__data_set

    @data_set.setter
    def data_set(self, dataset: str):
        self.__data_set = dataset

    @property
    def data_fold(self):
        return self.__data_fold

    @data_fold.setter
    def data_fold(self, datafold: str):
        self.__data_fold = datafold

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.__df

    @property
    def collate_fn(self) -> Optional[Callable]:
        return self.__collate_fn

    @collate_fn.setter
    def collate_fn(self, collate_fn: Optional[Callable]):
        self.__collate_fn = collate_fn

    def get_negative_sample_indices(self):
        return []

    def get_positive_sample_indices(self):
        return []

    def _get_sampler(self, dataset):
        return RandomSampler(dataset)

    def _get_id_column_name(self):
        return 'id'

    def get_label_column_name(self):
        return 'label'

    def _smooth_labels(self, labels):
        if self.__label_smoothing is None:
            return labels

        return labels * (1 - self.__label_smoothing) + self.__label_smoothing / self.__num_classes

    def _get_labels(self, record):
        return self._smooth_labels(record[self.get_label_column_name()].values.astype(np.float32))

    def _get_extras(self, record):
        return {}

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        return self.dataframe.shape[0]
