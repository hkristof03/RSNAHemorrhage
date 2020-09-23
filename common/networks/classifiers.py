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
import torch
# import cv2
# import random

from torch import Tensor
from typing import Optional, Tuple, List

from . import AbstractNetwork
# from config import DIR_EXPERIMENT_OUTPUT
from common.logger import DefaultLogger


class ImageClassifierNetwork(AbstractNetwork):

    def __init__(self, logger: Optional[DefaultLogger] = None, use_amp: bool = False, batch_size: int = 16,
                 accumulation_step: int = 1, iteration_num: int = 10, iteration_valid=100) -> None:
        super().__init__(logger, use_amp, batch_size, accumulation_step, iteration_num, iteration_valid)

    def _fit_batch(self, epoch: int, iteration: int, batch_idx: int,
                   batch_data: Tuple[Tensor, Tensor, Tensor, any]) -> Tuple[Tensor, Tensor]:

        indices, images, targets, extras = batch_data

        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()

        outputs = self.model(images)

        return outputs, targets

    def _fit_validation_batch(self, epoch: int, iteration: int, batch_idx: int,
                              batch_data: Tuple[Tensor, Tensor, Tensor, any]):
        indices, images, targets, extras = batch_data

        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()

        with torch.no_grad():
            outputs = self.model(images)

        return outputs, targets

    def _predict_batch(self, batch_idx: int, batch_data: Tuple[Tensor, Tensor, Tensor, any]) -> Tensor:

        indices, images, _, extras = batch_data

        """
        # DEBUG
        if batch_idx == 0:
            np_images = images.clone()
            np_images = np_images.data.cpu().numpy()
            np_images = np_images.transpose(0, 2, 3, 1)

            for i in range(len(np_images)):
                tmp_img = np_images[i]
                tmp_img = (tmp_img * 255).astype(np.uint8)
                cv2.imwrite(DIR_EXPERIMENT_OUTPUT + '/tmp_{}_{}.png'.format(i, random.randint(1000, 999999999)), tmp_img)
        """

        if torch.cuda.is_available():
            images = images.cuda()

        with torch.no_grad():
            outputs = self.model(images)

        return outputs

    def _compute_loss(self, outputs, target):
        raise NotImplementedError

