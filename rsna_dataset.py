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
import torch
import cv2
import pydicom
import importlib

from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from config import DIR_INPUT, DIR_TRANSFORMS, DIR_PROJECT
from common.datasets import ImageClassificationDataset, BalancedSampler, PositiveClassRatioSampler
from common.logger import DefaultLogger


class RSNADataset(ImageClassificationDataset):

    def __init__(self, df: pd.DataFrame, df_type: str, image_folder: str, logger: Optional[DefaultLogger] = None,
                 label_smoothing: Optional[float] = None) -> None:
        super().__init__(df, df_type, image_folder, logger, label_smoothing=label_smoothing)

    def _load_image(self, image_id):
        # return cv2.imread(self.image_folder + '/{}.png'.format(image_id), cv2.IMREAD_COLOR)
        dicom = pydicom.dcmread(self.image_folder + '/{}.dcm'.format(image_id))
        image = dicom.pixel_array

        image = self.rescale_image(image, dicom.RescaleSlope, dicom.RescaleIntercept)

        # img_r = self.apply_window(image, 40, 80)  # brain
        # img_g = self.apply_window(image, 80, 200)  # subdural
        # img_b = self.apply_window(image, 40, 380)  # bone
        # img_r = (img_r - 0) / 80
        # img_g = (img_g - (-20)) / 200
        # img_b = (img_b - (-150)) / 380
        # image = np.zeros((image.shape[0], image.shape[1], 3))
        # image[:, :, 0] = img_r  # - img_r.mean()
        # image[:, :, 1] = img_g  # - img_g.mean()
        # image[:, :, 2] = img_b  # - img_b.mean()
        # return image

        image3 = np.zeros((image.shape[0], image.shape[1], 3))
        image3[:, :, 0] = image
        image3[:, :, 1] = image
        image3[:, :, 2] = image

        return image3

    def apply_window(self, image, center, width):
        image = image.copy()

        min_value = center - width // 2
        max_value = center + width // 2

        image[image < min_value] = min_value
        image[image > max_value] = max_value

        return image

    def rescale_image(self, image, slope, intercept):
        return image * slope + intercept

    def _get_extras(self, record):
        return record[['id']].to_dict()

    def get_label_column_name(self):
        return ['cls_1', 'cls_2', 'cls_3', 'cls_4', 'cls_5', 'any']

    def get_negative_sample_indices(self):
        df = self.dataframe
        return df[df['any'] == 0].index.values

    def get_positive_sample_indices(self):
        df = self.dataframe
        return df[df['any'] == 1].index.values


class RSNAAnyBinaryDataset(RSNADataset):

    def __init__(self, df: pd.DataFrame, df_type: str, image_folder: str, logger: Optional[DefaultLogger] = None,
                 label_smoothing: Optional[float] = None) -> None:
        super().__init__(df, df_type, image_folder, logger, label_smoothing=label_smoothing)

    def get_label_column_name(self):
        return ['any']


def rsna_collate_fn(data):
    batch_size = len(data)

    indices = []
    images = []
    labels = []
    extras = []

    for b in range(batch_size):
        indices.append(data[b][0])
        # images.append(np.expand_dims(data[b][1], axis=0))
        images.append(data[b][1].transpose(2, 0, 1))
        labels.append(data[b][2])
        extras.append(data[b][3])

    indices = np.array(indices)

    images = np.stack(images)
    images = torch.from_numpy(images).float()

    labels = np.stack(labels)
    labels = torch.from_numpy(labels)

    return indices, images, labels, extras


def get_rsna_train_dataloader(args, logger=None, ds_class_name=None) -> DataLoader:
    dataset = args.data_dataset
    assert isinstance(dataset, str), "--data_dataset argument is missing"

    fold_id = args.data_fold
    assert fold_id >= 0, "--data_fold argument is missing"

    df = pd.read_csv(DIR_INPUT + '/datasets/{0}/dataset_{0}_fold_{1}.csv'.format(dataset, fold_id))
    df = df.replace(np.nan, '', regex=True)

    # HACK (majd meg kell csinálni rendesen)
    df.loc[df['id'] == 'ID_6431af929', 'sample_type'] = 'ignore'

    df = df[df['sample_type'] == 'train']
    df.reset_index(drop=True, inplace=True)

    if ds_class_name is not None:
        class_ = getattr(importlib.import_module("rsna_dataset"), ds_class_name)
        dataset = class_(df, 'train', image_folder=args.data_input_folder, logger=logger,
                         label_smoothing=args.tr_label_smoothing)

    else:
        dataset = RSNADataset(df, 'train', image_folder=args.data_input_folder, logger=logger)

    dataset.collate_fn = rsna_collate_fn
    dataset.data_set = args.data_dataset
    dataset.data_fold = args.data_fold

    if args.data_train_transform is not None:
        dataset.load_transforms(DIR_TRANSFORMS + '/' + args.data_train_transform)

    sampler = None

    if args.data_sampler is not None:

        if args.data_sampler == 'balanced':
            sampler = BalancedSampler(dataset)
        elif args.data_sampler == 'pos_range':
            logger.info("    sampler: PositiveClassRatioSamper")
            sampler = PositiveClassRatioSampler(dataset,
                                                num_iter=int(args.tr_iteration_num),
                                                batch_size=args.tr_batch_size,
                                                pos_range=(0.4, 0.14),
                                                update_at_every_iter=100)
        else:
            raise ValueError("Sampler `{}` is not implemented yet".format(args.data_sampler))

    return dataset.get_data_loader(
        batch_size=args.tr_batch_size,
        sampler=sampler,
        num_worker=args.n_jobs,
        pin_memory=False,
        drop_last=False
    )


def get_rsna_valid_dataloader(args, logger=None, frac=None, ds_class_name=None) -> DataLoader:
    dataset = args.data_dataset
    assert isinstance(dataset, str), "--data_dataset argument is missing"

    fold_id = args.data_fold
    assert fold_id >= 0, "--data_fold argument is missing"

    df = pd.read_csv(DIR_INPUT + '/datasets/{0}/dataset_{0}_fold_{1}.csv'.format(dataset, fold_id))
    df = df.replace(np.nan, '', regex=True)

    df = df[df['sample_type'] == 'valid']
    df.reset_index(drop=True, inplace=True)

    if frac is not None:
        df = df.sample(frac=frac)
        df.reset_index(drop=True, inplace=True)

    if ds_class_name is not None:
        class_ = getattr(importlib.import_module("rsna_dataset"), ds_class_name)
        dataset = class_(df, 'valid', image_folder=args.data_input_folder, logger=logger)
    else:
        dataset = RSNADataset(df, 'valid', image_folder=args.data_input_folder, logger=logger)

    dataset.collate_fn = rsna_collate_fn
    dataset.data_set = args.data_dataset
    dataset.data_fold = args.data_fold

    if args.data_valid_transform is not None:
        dataset.load_transforms(DIR_TRANSFORMS + '/' + args.data_valid_transform)

    return dataset.get_data_loader(
        batch_size=args.tr_batch_size,
        sampler=SequentialSampler(dataset),
        num_worker=args.n_jobs,
        pin_memory=False,
        drop_last=False
    )


def get_rsna_test_dataloader(args, logger=None, ds_class_name=None) -> DataLoader:
    df = pd.read_csv(DIR_PROJECT + '/' + args.pr_dataframe)

    dev = args.dev_enabled

    if dev:
        df = df.sample(n=64)
        df.reset_index(drop=True, inplace=True)

    if ds_class_name is not None:
        class_ = getattr(importlib.import_module("rsna_dataset"), ds_class_name)
        dataset = class_(df, 'test', image_folder=args.data_input_folder, logger=logger)
    else:
        dataset = RSNADataset(df, 'test', image_folder=args.data_input_folder, logger=logger)

    dataset.collate_fn = rsna_collate_fn

    if args.data_test_transform is not None:
        dataset.load_transforms(DIR_TRANSFORMS + '/' + args.data_test_transform)

    return dataset.get_data_loader(
        batch_size=args.tr_batch_size,
        sampler=SequentialSampler(dataset),
        num_worker=args.n_jobs,
        pin_memory=False,
        drop_last=False
    )
