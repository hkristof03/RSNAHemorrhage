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
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from typing import Optional

from common.logger import DefaultLogger
from common.optimizers.radam import RAdam
from common.optimizers.lookahead import Lookahead
from common.optimizers.lamb import Lamb
from common.optimizers.lars import Lars

try:
    from apex import amp
except ModuleNotFoundError:
    pass


def optimizer_factory(args, network, logger: Optional[DefaultLogger] = None):

    optim_type = args.optim
    lookahead_enabled = args.optim_lookahead_enabled
    amp_enabled = args.tr_use_amp
    model = network.model

    if optim_type is None:
        optim_type = 'sgd'

    optim_type = optim_type.lower()

    if logger is not None:
        logger.info("")
        logger.info("============== OPTIMIZER ==============")
        logger.info("    type = {}".format(optim_type))
        logger.info("    learning rate = {}".format(args.optim_lr))
        logger.info("    weight decay = {}".format(args.optim_weight_decay))

    if optim_type == 'sgd':
        optim = SGD(params=model.get_optimizer_params(), lr=args.optim_lr, momentum=args.optim_momentum,
                    weight_decay=args.optim_weight_decay, nesterov=args.optim_nesterov)

        if logger is not None:
            logger.info("    momentum = {}".format(args.optim_momentum))
            logger.info("    nesterov = {}".format(args.optim_nesterov))

    elif optim_type == 'adam':
        optim = Adam(params=model.get_optimizer_params(), lr=args.optim_lr, weight_decay=args.optim_weight_decay)

    elif optim_type == 'adamw':
        optim = AdamW(params=model.get_optimizer_params(), lr=args.optim_lr, weight_decay=args.optim_weight_decay)

    elif optim_type == 'radam':
        optim = RAdam(params=model.get_optimizer_params(), lr=args.optim_lr, weight_decay=args.optim_weight_decay)

    elif optim_type == 'lamb':
        optim = Lamb(params=model.get_optimizer_params(), lr=args.optim_lr, weight_decay=args.optim_weight_decay)

    elif optim_type == 'lars':
        optim = Lars(params=model.get_optimizer_params(), lr=args.optim_lr, momentum=args.optim_momentum,
                     weight_decay=args.optim_weight_decay, max_epoch=args.tr_iteration_num)

    else:
        raise NotImplementedError("The `{}` optimizer has not been implemented yet.".format(optim_type))

    if lookahead_enabled:
        optim = Lookahead(base_optimizer=optim, alpha=0.5, k=5)

    elif amp_enabled:
        network.model, optim = amp.initialize(model, optim,
                                              opt_level=args.tr_amp_level,
                                              keep_batchnorm_fp32=True if args.tr_use_amp == 'O2' else None,
                                              loss_scale='dynamic'
                                              )

    return optim
