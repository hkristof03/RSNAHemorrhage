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
import math

from common.logger import DefaultLogger
from common.schedulers import WarmupScheduler

from typing import Optional


class WarmupCosineSchedule(WarmupScheduler):
    """
    Linearly increases learning rate from 0 to max_lr over `warmup` fraction of training steps.
    Decreases learning rate from max_lr to min_lr. over remaining `1 - warmup` steps following a cosine curve.
    """
    warn_t_total = True

    def __init__(self, optimizer, max_lr: float = 1.0, min_lr: float = 0.0, warmup: float = 0.002, total_steps: int = -1,
                 logger: Optional[DefaultLogger] = None, **kw):

        self._min_lr = min_lr

        super().__init__(optimizer, max_lr, warmup, total_steps, logger, **kw)

    def get_lr_(self, progress):
        if progress < self.warmup:
            # return self._max_lr * (progress / self.warmup)
            return [self._max_lr * (progress / self.warmup) for _ in self.base_lrs]
        else:

            if progress > 1.0:
                progress = 1.0

            progress = (progress - self.warmup) / (1 - self.warmup)   # progress after warmup
            lr = (self._max_lr - self._min_lr) * math.cos(0.5 * math.pi * progress) + self._min_lr

            return [lr for _ in self.base_lrs]
