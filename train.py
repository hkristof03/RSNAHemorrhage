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
import neptune
import numpy as np
import sys
import warnings

from traceback import format_exception

from config import *
from common.logger import FileLogger
from common.callbacks import TrainingMonitor, FScoreMultilabel, MultilabelLogLoss, LearningRateMonitor, ModelCheckpoint
from common.optimizers import optimizer_factory
from common.schedulers import scheduler_factory
from rsna_dataset import get_rsna_train_dataloader, get_rsna_valid_dataloader
from rsna_network import RSNANetwork
from rsna_model import model_factory

np.seterr(divide="ignore")
warnings.filterwarnings("ignore")


def sev_excpetion_hook(exctype, value, tb):
    """Exception handler. A Runtime hibák logolásához"""

    rsna_logger.info("================= ERROR =================")

    trace = format_exception(exctype, value, tb)

    for msg in trace:
        rsna_logger.info("{}".format(msg.replace('\n', '')))

    rsna_logger.info("")
    rsna_logger.info("----- exiting -----")

    if nml and not dev:
        neptune.stop(tb)

    exit(1)


if __name__ == '__main__':

    cmd = " ".join(sys.argv)
    dev = args.dev_enabled
    nml = args.nml_enabled
    nml_exp = None
    sys.excepthook = sev_excpetion_hook
    rsna_logger = FileLogger('rsna', DIR_EXPERIMENT_OUTPUT, 'train', console=dev or not nml)

    if nml and not dev:
        rsna_logger.info("Initializing neptune.ml client...")

        # Init neptune
        neptune.init(NEPTUNE_PROJECT)
        nml_exp = neptune.create_experiment(name='EXP-' + str(args.experiment_id),
                                            logger=rsna_logger.logger,
                                            upload_stdout=False,
                                            tags=['dev'],
                                            params={
                                                'data_dataset': args.data_dataset,
                                                'data_fold': args.data_fold,
                                                'data_train_transform': args.data_train_transform if args.data_train_transform is not None else '-',
                                                'data_valid_transform': args.data_valid_transform if args.data_valid_transform is not None else '-',
                                                'data_sampler': args.data_sampler,

                                                'net_model': args.net_model,
                                                'net_loss': 'bce',
                                                'net_pretrained': args.net_pretrained,
                                                'net_weight_file': args.net_weight_file if args.net_weight_file is not None else '-',
                                                'net_num_classes': args.net_num_classes,

                                                'optim': args.optim,
                                                'optim_lr': args.optim_lr,
                                                'optim_momentum': args.optim_momentum,
                                                'optim_nesterov': args.optim_nesterov,
                                                'optim_weight_decay': args.optim_weight_decay,
                                                'optim_lookahead_enabled': args.optim_lookahead_enabled,
                                                'optim_scheduler': args.optim_scheduler,
                                                'optim_scheduler_max_lr': args.optim_scheduler_max_lr,

                                                'tr_iteration_num': args.tr_iteration_num,
                                                'tr_batch_size': args.tr_batch_size,
                                                'tr_accumulation_step': args.tr_accumulation_step
                                            },
                                            properties={
                                                'command': cmd,
                                            },
                                            upload_source_files=[
                                                'config.py',
                                                'predict.py',
                                                'predict.sh',
                                                'preprocessing.py'
                                                'rsna_dataset.py',
                                                'rsna_model.py',
                                                'rsna_network.py',
                                                'train.py',
                                                'train.sh',
                                                'common/**/*.py',
                                                'transforms/*.json',
                                            ])
        rsna_logger.neptune_experiment = nml_exp

    rsna_logger.info('======================================================')
    rsna_logger.info('')
    rsna_logger.info('                       TRAIN                         ')
    rsna_logger.info('')
    rsna_logger.info('======================================================')
    rsna_logger.info('')
    rsna_logger.info(cmd)
    rsna_logger.info('')

    if dev:
        rsna_logger.info('######################################################')
        rsna_logger.info('         !!! DEVELOPER MODE ENABLED !!!               ')
        rsna_logger.info('######################################################')
        rsna_logger.info('')
        rsna_logger.info('')

    rsna_logger.info('------------------------------------------------------')
    rsna_logger.info('                 CONFIGURATION                        ')
    rsna_logger.info('------------------------------------------------------')
    rsna_logger.info('')

    for arg in vars(args):
        rsna_logger.info('{} = {}'.format(arg, getattr(args, arg)))

    rsna_logger.info('')

    network = RSNANetwork(logger=rsna_logger,
                          use_amp=args.tr_use_amp,
                          batch_size=args.tr_batch_size,
                          accumulation_step=args.tr_accumulation_step,
                          iteration_num=args.tr_iteration_num,
                          iteration_valid=args.tr_iteration_valid)

    network.add_callback(TrainingMonitor(logger=rsna_logger, log_every_iteration=10))
    network.add_callback(LearningRateMonitor(logger=rsna_logger, log_every_iteration=10))

    network.add_callback(MultilabelLogLoss(logger=rsna_logger, save_best=True, output_dir=DIR_EXPERIMENT_OUTPUT,
                                           class_names=['Epidural(1)', 'Intraparenchymal(2)', 'Intraventricular(3)',
                                                        'Subarachnoid(4)', 'Subdural(5)', 'Any(0)'],
                                           log_iteration=args.tr_iteration_valid))

    # network.add_callback(ModelCheckpoint(logger=rsna_logger,
    #                                      best_in_iter=round(args.tr_iteration_num // 16, -3),
    #                                      output_dir=DIR_EXPERIMENT_OUTPUT))

    network.add_dataloader('train', get_rsna_train_dataloader(args, logger=rsna_logger))
    network.add_dataloader('validation', get_rsna_valid_dataloader(args, logger=rsna_logger))

    network.model = model_factory(args, logger=rsna_logger, weight_file=args.net_weight_file)
    network.optimizer = optimizer_factory(args, network=network, logger=rsna_logger)
    network.scheduler = scheduler_factory(args, network.optimizer, logger=rsna_logger)

    network.fit()
