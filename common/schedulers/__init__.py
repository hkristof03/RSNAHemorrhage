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
from torch.optim.lr_scheduler import MultiStepLR
from .one_cycle_lr import OneCycleLR
from .schedulers import WarmupScheduler
from .warmup_cosine import WarmupCosineSchedule


def scheduler_factory(args, optimizer, logger=None):

    scheduler = args.optim_scheduler

    if scheduler is None:
        return None

    scheduler = scheduler.lower()

    if logger is not None:
        logger.info("")
        logger.info("============== SCHEDULER ==============")
        logger.info("    type = {}".format(scheduler))
        logger.info("    max_lr = {}".format(args.optim_scheduler_max_lr))
        logger.info("    total_steps = {}".format(args.tr_iteration_num))

    if scheduler == 'one_cycle_lr':
        scheduler = OneCycleLR(optimizer,
                               max_lr=args.optim_scheduler_max_lr,
                               total_steps=args.tr_iteration_num,
                               pct_start=args.optim_scheduler_pct_start)

        if logger is not None:
            logger.info("    pct_start = {}".format(args.optim_scheduler_pct_start))

    elif scheduler == 'warmup_cosine':
        scheduler = WarmupCosineSchedule(optimizer,
                                         max_lr=args.optim_scheduler_max_lr,
                                         min_lr=args.optim_scheduler_min_lr,
                                         total_steps=args.tr_iteration_num,
                                         warmup=args.optim_scheduler_warmup,
                                         logger=logger)
        if logger is not None:
            logger.info("    max_lr = {}".format(args.optim_scheduler_min_lr))

    elif scheduler == 'multistep':
        scheduler = MultiStepLR(optimizer,
                                milestones=(40000, 80000, 115000),
                                gamma=args.optim_scheduler_gamma)

    else:
        raise NotImplementedError("The `{}` scheduler has not been implemented yet".format(scheduler))

    return scheduler

