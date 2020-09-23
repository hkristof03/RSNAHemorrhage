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
import collections
import numpy as np
import pandas as pd
import random

from pprint import pprint
from tqdm import tqdm


def _make_folds(dataframe, n_fold, seed):
    counter_gt = collections.defaultdict(int)
    for labels in ['any', 'cls_1', 'cls_2', 'cls_3', 'cls_4', 'cls_5']:
        for label in labels:
            counter_gt[label] += 1

    counter_folds = collections.Counter()

    _folds = {}
    random.seed(seed)
    groups = dataframe.groupby('PatientID')
    print('making %d _folds...' % n_fold)
    for patient_id, group in tqdm(groups, total=len(groups)):

        labels = []
        for row in group.itertuples():
            for label in ['any', 'cls_1', 'cls_2', 'cls_3', 'cls_4', 'cls_5']:
                labels.append(label)
        if not labels:
            labels = ['']

        count_labels = [counter_gt[label] for label in labels]
        min_label = labels[np.argmin(count_labels)]
        count_folds = [(f, counter_folds[(f, min_label)]) for f in range(n_fold)]
        min_count = min([count for f, count in count_folds])
        fold = random.choice([f for f, count in count_folds if count == min_count])
        _folds[patient_id] = fold

        for label in labels:
            counter_folds[(fold, label)] += 1

    pprint(counter_folds)

    return _folds


if __name__ == '__main__':

    NUM_FOLDS = 5
    SEED = 4667

    df = pd.read_csv('input2/preprocessed_c_dicom/stage_1_train.csv')
    folds = _make_folds(df, NUM_FOLDS, SEED)

    df['fold'] = df.PatientID.map(folds)
    df['sample_type'] = 'train'

    for fold_id in range(NUM_FOLDS):
        tmp_df = df.copy()
        tmp_df.loc[tmp_df['fold'] == fold_id, 'sample_type'] = 'valid'

        tmp_df.to_csv('input/datasets/c/dataset_c_fold_{}.csv'.format(fold_id), index=False)

