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


def rotate90(batch, k):
    """rotate(images, degree) -> Tensor

    Rotates the images

    Args:
        batch (tensor): Input tensor (batch of images), the shape should be N x C x H x W, where
            - N - Number of images in the batch
            - C - Number of channels
            - H - Image height
            - W - Image width

        k (int) number of times to rotate
    """
    return torch.rot90(batch, k=k, dims=(3, 2))


def flip_h(batch):
    """flip_h(batch) -> Tensor

    Flips the images horizontally

    Args:
        batch (tensor): Input tensor (batch of images), the shape should be N x C x H x W, where
            - N - Number of images in the batch
            - C - Number of channels
            - H - Image height
            - W - Image width
    """
    return torch.flip(batch, dims=(2, ))


def flip_v(batch):
    """flip_v(batch) -> Tensor

    Flips the images vertically

    Args:
        batch (tensor): Input tensor (batch of images), the shape should be N x C x H x W, where
            - N - Number of images in the batch
            - C - Number of channels
            - H - Image height
            - W - Image width
    """
    return torch.flip(batch, dims=(3, ))
