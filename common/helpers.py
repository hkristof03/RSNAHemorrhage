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
import os
import yaml
import numpy as np
import random as rn
import torch
import torch.nn as nn

from functools import partial
from typing import Iterable, Callable, Optional


def load_from_yaml(file):
    if not os.path.isfile(file):
        return {
            'id': 0
        }

    with open(file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return {'id': 0}


def save_to_yaml(file, data):
    with open(file, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def init_seeds(seed):
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(seed)

    # PyTorch seeds
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    # Other
    # os.environ['PYTHONHASHSEED'] = str(seed)


def persist_torch_model(file, epoch, iteration, model_state_dict, optimizer_state_dict=None):
    data = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': None
    }

    torch.save(data, file)


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class MultiAverager:
    def __init__(self):
        self.current_total = None
        self.iterations = None

    def send(self, value, iterations=0):

        if self.current_total is None:
            self.current_total = value
            self.iterations = iterations

        self.current_total += value
        self.iterations += iterations

    def value(self):
        if self.iterations is None:
            return 0
        else:
            if self.iterations == 0:
                self.iterations = 1

            return 1.0 * self.current_total / self.iterations

    def reset(self, totals=None, iterations=None):
        self.current_total = totals
        self.iterations = iterations


def listify(o):
    if o is None:
        return []

    elif isinstance(o, list):
        return o

    elif isinstance(o, str):
        return [o]

    elif isinstance(o, Iterable):
        return list(o)

    return [o]


# From Fast.AI course v3
class ListContainer:

    def __init__(self, items) -> None:
        self.items = listify(items)

    def __getitem__(self, idx):

        if isinstance(idx, (int, slice)):
            return self.items[idx]

        elif isinstance(idx[0], bool):
            assert len(idx) == len(self)

            return [items for item_idx, items in zip(idx, self.items) if item_idx]

        return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, index, value):
        self.items[index] = value

    def __delitem__(self, index):
        del(self.items[index])

    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items\n{self.items[:10]})'

        if len(self) > 10:
            res = '[' + res[:-1] + '...]'

        return res


class Hook:

    def __init__(self, module: nn.Module, forward_callback: Callable) -> None:
        super().__init__()

        self.hook = module.register_forward_hook(partial(forward_callback, self))

    def removeHook(self):
        self.hook.remove()

    def __del__(self):
        self.removeHook()


class Hooks(ListContainer):

    def __init__(self, modules: Iterable, hook_cls, forward_callback: Optional[Callable] = None) -> None:
        super().__init__([hook_cls(module, forward_callback) for module in modules])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, index):
        self[index].remove()
        super().__delitem__(index)

    def remove(self):
        for hook in self:
            hook.remove()
