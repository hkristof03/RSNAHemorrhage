# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""
import warnings

from common.logger import DefaultLogger
from torch.optim.optimizer import Optimizer
from functools import partial, wraps
from typing import Optional


class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1):

        if not isinstance(optimizer, Optimizer):

            # Try APEX
            flag = False
            try:
                from apex.fp16_utils.fp16_optimizer import FP16_Optimizer
                if isinstance(optimizer, FP16_Optimizer):
                    flag = True

            except ModuleNotFoundError:
                pass

            if not flag:
                raise TypeError('{} is not an Optimizer'.format(
                    type(optimizer).__name__))

        self.optimizer = optimizer

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(func, opt):
            @wraps(func)
            def wrapper(*args, **kwargs):
                opt._step_count += 1
                return func(*args, **kwargs)
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step, self.optimizer)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step(last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule."
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class WarmupScheduler(_LRScheduler):
    """ Parent of all LRSchedules here. """
    warn_t_total = False        # is set to True for schedules where progressing beyond t_total steps doesn't make sense

    def __init__(self, optimizer, max_lr: float = 1.0, warmup: float = 0.002, total_steps: int = -1,
                 logger: Optional[DefaultLogger] = None, **kw):
        """
        :param max_lr:  Maximum LR
        :param warmup:  what fraction of t_total steps will be used for linear warmup
        :param total_steps: how many training steps (updates) are planned
        :param logger:  Logger
        :param kw:
        """
        self._logger = logger
        self._max_lr = max_lr

        if total_steps < 0:
            self._logger.warning("t_total value of {} results in schedule not being applied".format(total_steps))

        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))

        warmup = max(warmup, 0.)

        self.warmup, self.total_steps = float(warmup), float(total_steps)
        self.warned_for_t_total_at_progress = -1

        super(WarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        """
        :param step:    which of t_total steps we're on
        :param nowarn:  set to True to suppress warning regarding training beyond specified 't_total' steps
        :return:        learning rate multiplier for current update
        """
        if self.total_steps < 0:
            return self._max_lr

        progress = float(self._step_count) / self.total_steps
        ret = self.get_lr_(progress)

        # warning for exceeding t_total (only active with warmup_linear
        # if not nowarn and self.warn_t_total and progress > 1. and progress > self.warned_for_t_total_at_progress:
        #     self.__logger.warning("Training beyond specified 't_total'. Learning rate multiplier set to {}. "
        #                           "Please set 't_total' of {} correctly.".format(ret, self.__class__.__name__))
        #     self.warned_for_t_total_at_progress = progress
        # end warning

        return ret

    def get_lr_(self, progress):
        """
        :param progress:    value between 0 and 1 (unless going beyond t_total steps) specifying training progress
        :return:            learning rate multiplier for current update
        """
        return [self._max_lr for _ in self.base_lrs]
