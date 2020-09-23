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
import pandas as pd
import warnings

from config import *
from common.logger import FileLogger

np.seterr(divide="ignore")
warnings.filterwarnings("ignore")

SUBMISSION_FILES = [
    '/468/preds___467_ds_c__fold_0__best_log_loss__iter_69499__0_0902848.csv',
    '/483/preds___478_ds_c__fold_0__best_log_loss__iter_185499__0_0850694.csv',
    '/505/preds___486_ds_c__fold_0__best_log_loss__iter_130499__0_0881273.csv'
]

if __name__ == '__main__':

    rsna_logger = FileLogger('rsna', DIR_EXPERIMENT_OUTPUT, 'train', console=True)

    rsna_logger.info('======================================================')
    rsna_logger.info('')
    rsna_logger.info('                       ENSEMBLE                        ')
    rsna_logger.info('')
    rsna_logger.info('======================================================')

    rsna_logger.info('')

    rsna_logger.info('------------------------------------------------------')
    rsna_logger.info('                 PREDICTIONS                          ')
    rsna_logger.info('------------------------------------------------------')
    rsna_logger.info('')

    for f in SUBMISSION_FILES:
        rsna_logger.info(('    {}'.format(f)))

    rsna_logger.info('')

    df = pd.read_csv(DIR_RESULTS + SUBMISSION_FILES[0])
    df.rename(columns={'Label': 'Label_0'}, inplace=True)

    for idx in range(1, len(SUBMISSION_FILES)):
        edf = pd.read_csv(DIR_RESULTS + SUBMISSION_FILES[idx])
        edf.rename(columns={'Label': 'Label_{}'.format(idx)}, inplace=True)
        df = df.merge(edf, how='left', on=['ID'])

    df['mean'] = df.iloc[:, 1:].mean(axis=1)
    df = df[['ID', 'mean']]
    df.rename(columns={'mean': 'Label'}, inplace=True)

    df.to_csv(DIR_EXPERIMENT_OUTPUT + '/ensembled__{}_submission.csv'.format(EXPERIMENT_ID), index=False)
