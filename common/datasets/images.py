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
import cv2

from albumentations import Compose
from albumentations import load as albu_load
from typing import Optional, Callable

from . import AbstractDataset
from common.logger import DefaultLogger
from common.albumentations import DicomWindowShift


class ImageClassificationDataset(AbstractDataset):

    def __init__(self, df: pd.DataFrame, df_type: str, image_folder: str, logger: Optional[DefaultLogger] = None,
                 label_smoothing: Optional[float] = None) -> None:
        super().__init__(df, df_type, logger, label_smoothing=label_smoothing)

        self.__image_folder = image_folder
        self.__transforms = None

        self._info("    image_folder = {}".format(self.image_folder))

    @property
    def transforms(self) -> Compose:
        return self.__transforms

    @transforms.setter
    def transforms(self, transforms: Compose):
        self.__transforms = transforms

    @property
    def image_folder(self) -> str:
        return self.__image_folder

    def load_transforms(self, transform_config) -> None:
        if isinstance(transform_config, str):
            self.transforms = albu_load(transform_config)
        else:
            self.transforms = transform_config

    def _load_image(self, image_id):

        if image_id[:-4].lower() not in ['.png', '.jpg']:
            raise ValueError("The `image_id` should ends with one of [.png|.jpg|.jpeg]")

        return cv2.imread(self.image_folder + '/' + image_id, cv2.IMREAD_COLOR)

    def _apply_transforms(self, image_np):
        if self.transforms:
            sample = {'image': image_np}
            sample = self.transforms(**sample)

            image_np = sample['image']

        return image_np

    def __getitem__(self, index: int):
        """Returns the image selected by a `Sampler`"""

        # Row from pd.DataFrame
        record = self.dataframe.iloc[index]

        # Labels
        labels = self._get_labels(record)

        # Extras
        extras = self._get_extras(record)

        # Numpy array
        image_np = self._load_image(record[self._get_id_column_name()])

        if self.transforms is not None:
            image_np = self._apply_transforms(image_np)

        return index, image_np, labels, extras


class ImageSegmentationDataset(ImageClassificationDataset):

    def __init__(self, df: pd.DataFrame, df_type: str, image_folder: str, mask_folder: str  = None,
                 logger: Optional[DefaultLogger] = None) -> None:
        super().__init__(df, df_type, image_folder, logger)

        self.__mask_folder = mask_folder

