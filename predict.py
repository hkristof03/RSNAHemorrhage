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
import sys
import warnings

from config import *
from common.logger import FileLogger
from common.callbacks import TrainingMonitor

from rsna_dataset import get_rsna_test_dataloader
from rsna_network import RSNANetwork
from rsna_model import model_factory

np.seterr(divide="ignore")
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    cmd = " ".join(sys.argv)
    dev = args.dev_enabled
    nml = args.nml_enabled
    nml_exp = None

    rsna_logger = FileLogger('rsna', DIR_EXPERIMENT_OUTPUT, 'train', console=dev or not nml)

    rsna_logger.info('======================================================')
    rsna_logger.info('')
    rsna_logger.info('                       PREDICT                         ')
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
                          batch_size=args.tr_batch_size)

    network.add_dataloader('test', get_rsna_test_dataloader(args, logger=rsna_logger))
    network.model = model_factory(args, logger=rsna_logger, weight_file=args.net_weight_file)

    weight_file_id = args.net_weight_file.replace('./output/results', '').replace('/', '_').replace('.pt', '')

    if args.pr_tta > 0:

        predictions = None

        for tta in range(args.pr_tta):
            tta_predictions = network.predict()

            if predictions is None:
                predictions = tta_predictions
            else:
                predictions['Label_tta_{}'.format(tta)] = tta_predictions['Label'].values

        predictions['mean'] = predictions.iloc[:, 1:].mean(axis=1)

        predictions.to_csv(DIR_EXPERIMENT_OUTPUT + '/preds_tta__{}.csv'.format(weight_file_id), index=False)

        predictions = predictions[['ID', 'mean']]
        predictions = predictions.rename(columns={'mean': 'Label'})

    else:
        predictions = network.predict()

    predictions.to_csv(DIR_EXPERIMENT_OUTPUT + '/preds__{}.csv'.format(weight_file_id), index=False)
