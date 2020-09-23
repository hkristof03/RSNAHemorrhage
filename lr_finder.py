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
import sys
import torch
import torch.nn.functional as F
import warnings

from config import *
from common.logger import FileLogger
from common.optimizers import optimizer_factory
from common.misc import LRFinder

from rsna_dataset import get_rsna_train_dataloader, get_rsna_valid_dataloader
from rsna_model import model_factory


def criterion(outputs, target):
    batch_size, num_class, H, W = outputs.shape

    logit = outputs.view(batch_size, num_class)
    truth = target.view(batch_size, num_class)

    assert (logit.shape == truth.shape)

    # Normal sampler
    # weights = torch.tensor([243, 19, 27, 19, 14, 5]).float().cuda()
    weights = None

    loss = F.binary_cross_entropy_with_logits(logit, truth, pos_weight=weights)

    return loss


if __name__ == '__main__':

    cmd = " ".join(sys.argv)
    dev = args.dev_enabled
    nml = args.nml_enabled
    nml_exp = None
    rsna_logger = FileLogger('rsna', DIR_EXPERIMENT_OUTPUT, 'train', console=dev or not nml)

    train_loader = get_rsna_train_dataloader(args, logger=rsna_logger)
    valid_loader = get_rsna_valid_dataloader(args, logger=rsna_logger, frac=0.1)
    model = model_factory(args, logger=rsna_logger)
    optimizer = optimizer_factory(args, model=model, logger=rsna_logger)

    lr_finder = LRFinder(model, optimizer, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')
    lr_finder.range_test(train_loader, valid_loader, end_lr=1e3, num_iter=200, step_mode='exp')

    # print(lr_finder.history)

    lr_finder.plot()
