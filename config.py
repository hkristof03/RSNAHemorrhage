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
import argparse
import os
import torch

# torch.autograd.set_detect_anomaly(True)

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

from common.helpers import load_from_yaml, save_to_yaml, init_seeds

DIR_PROJECT = os.path.dirname(os.path.realpath(__file__))
DIR_INPUT = DIR_PROJECT + '/input'
DIR_OUTPUT = DIR_PROJECT + '/output'
DIR_CACHE = DIR_PROJECT + '/cache'
DIR_RESULTS = DIR_OUTPUT + '/results'
DIR_TRANSFORMS = DIR_PROJECT + '/transforms'

if not os.path.exists(DIR_CACHE):
    os.mkdir(DIR_CACHE)

if not os.path.exists(DIR_RESULTS):
    os.mkdir(DIR_RESULTS)

if not os.path.isfile(DIR_CACHE + '/experiments.yaml'):
    experiment = {
        'id': 0,
        'neptune_project': False
    }
    save_to_yaml(DIR_CACHE + '/experiments.yaml', experiment)

experiment = load_from_yaml(DIR_CACHE + '/experiments.yaml')
experiment['id'] += 1
save_to_yaml(DIR_CACHE + '/experiments.yaml', experiment)

EXPERIMENT_ID = experiment['id']
NEPTUNE_PROJECT = experiment['neptune_project']

DIR_EXPERIMENT_OUTPUT = DIR_RESULTS + '/' + str(EXPERIMENT_ID)

if not os.path.exists(DIR_EXPERIMENT_OUTPUT):
    os.mkdir(DIR_EXPERIMENT_OUTPUT)

parser = argparse.ArgumentParser()
arg = parser.add_argument


# ======================================
# General arguments
arg('--experiment_id', default=EXPERIMENT_ID,
    help='id of the experiment for crating unique files, folders, etc.')
arg('--seed', type=int, default=4667, help='seed for random generators')
arg('--dev_enabled', type=int, default=0)
arg('--n_jobs', type=int, default=1, help='number of jobs for parallel processing')

arg('--output_dir', type=str, default=DIR_OUTPUT, help='Output folder for the experiment. It is relative to the'
                                                       ' project\'s root folder and has to start with /')

# ======================================
# Data arguments
arg('--data_dataset', type=str, default='a')
arg('--data_fold', type=int, default=-1, help='Fold id')
arg('--data_input_folder', type=str, default='./input')
arg('--data_train_transform', type=str, default=None, help='Serialized albumentations transofrms (as .json)')
arg('--data_valid_transform', type=str, default=None, help='Serialized albumentations transofrms (as .json)')
arg('--data_test_transform', type=str, default=None,
    help='Serialized albumentations transforms for predict/tta (as .json)')
arg('--data_sampler', type=str, default=None)
arg('--data_img_size', type=int, default=256)

# ======================================
# Network arguments
arg('--net_model', type=str)
arg('--net_pretrained', type=int, default=1, help='Backbones pretrained weights')
arg('--net_weight_file', type=str, default=None, help='Pretrained weights for the network')
arg('--net_num_classes', type=int, default=6, help='Number of output classes')

# ======================================
# Optimizer arguments
arg('--optim', type=str, default='SGD', help='Optimizer')
arg('--optim_lr', type=float, default=0.01)
arg('--optim_momentum', type=float, default=0.9)
arg('--optim_nesterov', type=int, default=0)
arg('--optim_weight_decay', type=float, default=0.0001)
arg('--optim_lookahead_enabled', type=int, default=0)
arg('--optim_scheduler', type=str, default=None)
arg('--optim_scheduler_max_lr', type=float, default=0.01)
arg('--optim_scheduler_min_lr', type=float, default=0.0)
arg('--optim_scheduler_pct_start', type=float, default=0.3)
arg('--optim_scheduler_warmup', type=float, default=0.1)
arg('--optim_scheduler_gamma', type=float, default=1.0)
arg('--optim_swa_enabled', type=int, default=0, help='Enable Stochastic Weight Averaging')
arg('--optim_swa_start', type=int, default=50)
arg('--optim_swa_lr', type=float, default=0.05)

# ======================================
# Training arguments
arg('--tr_use_amp', type=int, default=0, help='Use APEX mixed precision training (need compatible model/optim)')
arg('--tr_amp_level', type=str, default='O0')
# 1 epoch = #num of samples // batch_size + 1
arg('--tr_iteration_num', type=int, default=10, help='Number of training iterations (batches pass to forward)')
arg('--tr_batch_size', type=int, default=16, help='Batch size for training')
arg('--tr_accumulation_step', type=int, default=1)
arg('--tr_iteration_valid', type=int, default=100, help='Run validation after number of batches')
arg('--tr_label_smoothing', type=float, default=None)

# ======================================
# Prediction arguments
arg('--pr_dataframe', type=str, default='./sample_submission.csv')
arg('--pr_tta', type=int, default=0, help='Number of test time augmentations')

# ======================================
# Telemetry Callback arguments
arg('--cb_telemetry_enabled', type=int, default=0, help='Enable telemetry callback')
arg('--cb_telemetry_module_name', type=str, default=None)
arg('--cb_telemetry_log_every', type=int, default=500, help='Update telemetry data after every x iterations')

# ======================================
# NEPTUNE
arg('--nml_enabled', type=int, default=0, help='Enable neptune.ml monitoring')

args = parser.parse_args()

if not experiment['neptune_project']:
    args.nml_enabled = 0

if not APEX_AVAILABLE:
    args.tr_use_amp = 0

# Reset random seed
init_seeds(args.seed)
