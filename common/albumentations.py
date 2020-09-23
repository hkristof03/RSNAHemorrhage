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

from albumentations.core.transforms_interface import ImageOnlyTransform


def apply_window(image, center, width):
    image = image.copy()

    min_value = center - width // 2
    max_value = center + width // 2

    image[image < min_value] = min_value
    image[image > max_value] = max_value

    return image


def dicom_window_shift(img, windows, min_max_normalize=True):
    image = np.zeros((img.shape[0], img.shape[1], 3))

    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    for i in range(3):
        ch = apply_window(img[:, :, i], windows[i][0], windows[i][1])

        if min_max_normalize:
            image[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-15)
        else:
            image[:, :, i] = ch

    return image


class DicomWindowShift(ImageOnlyTransform):

    def __init__(
            self,
            window_width_mins=(80, 200, 380),
            window_width_maxs=(80, 200, 380),
            window_center_mins=(40, 80, 40),
            window_center_maxs=(40, 80, 40),
            min_max_normalize=True,
            always_apply=False,
            p=0.5,
    ):
        super(DicomWindowShift, self).__init__(always_apply, p)
        self.window_width_mins = window_width_mins
        self.window_width_maxs = window_width_maxs
        self.window_center_mins = window_center_mins
        self.window_center_maxs = window_center_maxs
        self.min_max_normalize = min_max_normalize

        assert len(self.window_width_mins) == 3
        assert len(self.window_width_maxs) == 3
        assert len(self.window_center_mins) == 3
        assert len(self.window_center_maxs) == 3

    def apply(self, image, windows=(), min_max_normalize=True, **params):
        return dicom_window_shift(image, windows, min_max_normalize)

    def get_params_dependent_on_targets(self, params):
        windows = []

        for i in range(3):
            window_width = random.randint(self.window_width_mins[i], self.window_width_maxs[i])
            window_center = random.randint(self.window_center_mins[i], self.window_center_maxs[i])

            windows.append([window_center, window_width])

        return {"windows": windows, "min_max_normalize": self.min_max_normalize}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "window_width_mins", "window_width_maxs", "window_center_mins", "window_center_maxs", "min_max_normalize"
