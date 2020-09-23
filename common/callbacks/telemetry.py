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
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import Tensor

from common.callbacks import Callback
from common.helpers import Hooks, Hook
from common.logger import DefaultLogger
from typing import Callable


class TelemetryHook(Hook):

    def __init__(self, module: nn.Module, forward_callback: Callable = None) -> None:
        super().__init__(module, forward_callback if forward_callback is not None else self.hook_fn)

        self.__means = []
        self.__stds = []
        self.__hists = []
        self.__module_name = module.__class__.__name__

    def hook_fn(self, hook: Hook, module: nn.Module, inp: Tensor, out: Tensor, *args, **kwargs) -> None:
        # print(module)
        # print(inp[0].shape)
        # print(out.shape)
        if module.training:
            self.__means.append(out.data.mean())
            self.__stds.append(out.data.std())
            self.__hists.append(out.data.cpu().histc(40, -7, 7))

    def reset(self):
        self.__means = []
        self.__stds = []
        self.__hists = []

    @property
    def module_name(self):
        return self.__module_name

    @property
    def means(self):
        return self.__means

    @property
    def hists(self):
        return torch.stack(self.__hists).t().float().log1p()

    @property
    def hists_min(self):
        h1 = torch.stack(self.__hists).t().float()
        return h1[19:22].sum(0) / h1.sum(0)

    @property
    def stds(self):
        return self.__stds


class Telemetry(Callback):

    def __init__(self, model: nn.Module, module_name: str = '', logger: DefaultLogger = None,
                 log_every_iteration: int = 50, output_dir='.') -> None:
        super().__init__(logger)

        self._info("")
        self._info("-------------- TELEMETRY --------------")
        self._info("    modules = {}".format(module_name))
        self._info("    log_every_iteration = {}".format(log_every_iteration))

        self.__log_every_iteration = log_every_iteration
        self.__current_iteration = -1
        self.__output_dir = output_dir
        self.__hooks = None

        self.__register_hooks(model, module_name)

    def on_iteration_end(self, epoch: int, iteration: int, batch_idx: int, batch_data: any, outputs: Tensor,
                         targets: Tensor, loss: Tensor, *args, **kwargs):
        super().on_iteration_end(epoch, iteration, batch_idx, batch_data, outputs, targets, loss, *args, **kwargs)

        if (iteration + 1) % self.__log_every_iteration == 0:

            for i, hook in enumerate(self.__hooks):
                plt.plot(hook.means)
                plt.savefig(self.__output_dir + '/{}_means_{}.png'.format(hook.module_name, i))
                plt.close()

                plt.plot(hook.stds)
                plt.savefig(self.__output_dir + '/{}_stds_{}.png'.format(hook.module_name, i))
                plt.close()

                plt.imshow(hook.hists, cmap='afmhot', origin='lower')
                plt.savefig(self.__output_dir + '/{}_hists_{}.png'.format(hook.module_name, i))
                plt.close()

                plt.plot(hook.hists_min)
                plt.ylim(bottom=0.0, top=1.0)
                plt.savefig(self.__output_dir + '/{}_hists_min_{}.png'.format(hook.module_name, i))
                plt.close()

                hook.reset()

    def on_train_end(self, *args, **kwargs):
        self.__hooks.remove()

    def __register_hooks(self, model: nn.Module, module_name: str = ''):
        """
        Register a forward hook for every child modules of `module_name`.

        You can pass nested names like `encoder._blocks`

        TODO:
        You can also pass list of names (or list of nested names) sperated by comma `encoder._blocks,_fc`

        :param model:
        :param module_name:
        :return:
        """
        if module_name == '':
            return

        module_instance = model
        sub_modules = module_name.split('.')

        for sub_module in sub_modules:
            module_instance = getattr(module_instance, sub_module)

        self._info("Registering telemetry hooks...")
        self.__hooks = Hooks(module_instance, TelemetryHook)
