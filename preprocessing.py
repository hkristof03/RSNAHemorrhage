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
import sys
import traceback
import numpy as np
import pandas as pd
import pydicom
import scipy.ndimage
import cv2
import functools

from skimage import measure
from tqdm import tqdm
from multiprocessing import Pool, log_to_stderr


def process_helper(i, row_id, new_row):
    return rsna_converter.process(i, row_id, new_row)


class RSNADatasetConverter:

    def __init__(self, source_dataframe, input_folder, stage=1, dataset='train',
                 output_folder='./rsna_preprocessed', save_dicom_metadata=True, save_pixel_stats=True,
                 target_pixel_spacing=(0.5, 0.5), mask_head=True, crop_head=True, target_size=(256, 256),
                 normalize=(-750, 2000), zero_center=0.3, window=(0, 0), channels=False,
                 remove_dicom_files=False, debug=False, n_jobs=-1, ouput_type='npz'
                 ) -> None:

        super().__init__()

        # Save DICOM metadata to the output csv
        self.__save_dicom_metadata = save_dicom_metadata

        # Save output image pixel stats (basic stats: mean, min, max, etc.) to the csv
        self.__save_pixel_stats = save_pixel_stats

        # HU window to use (center, width)
        self.__window = window

        # Target pixel spacing for resampling
        self.__target_pixel_spacing = target_pixel_spacing

        # Mask the 'head' and remove everything else
        self.__mask_head = mask_head | crop_head

        # Crop the 'head' and resize the image based on the mask
        self.__crop_head = crop_head

        # Output size of the image. For downscale we use resize, for upscale we use padding
        self.__target_size = target_size

        # Delete the source file (for saving space)
        self.__remove_dicom_files = remove_dicom_files

        # Source dataframe (.csv)
        self.__source_dataframe = source_dataframe

        # Normalize (min, max)
        self.__normalize = normalize

        # Mean pixel values after normalization (all images)
        self.__zero_center = zero_center

        # 3 HU window convert to r-g-b channels
        self.__channels = channels

        # Stage
        self.__stage = stage

        # Dataset [`train` | `test`]
        self.__dataset = dataset

        # Input/Output folder
        self.__input_folder = input_folder
        self.__output_folder = output_folder
        self.__output_folder_image = self.__output_folder + '/stage_{}_{}_images'.format(stage, dataset)
        self.__output_type = ouput_type

        # Output dataframe
        self.__df_name = 'stage_{}_{}.csv'.format(stage, dataset)
        self.__df = None
        self.__data = []
        self.__processed = []

        self.__debug = debug
        self.__n_jobs = n_jobs

        self.__pbar = None
        self.__pool = None

        # Create output folder
        os.makedirs(self.__output_folder_image, exist_ok=True)

    def convert(self):
        """DICOM - numpy array konvertálás

        Több szálon futtatja a konvertáló metódust, hogy gyorsabban végezzen.

        BUG: a checkpoint mentés/betöltsé RuntimeError hibák eseten nem működik
        """
        # self.__load_checkpoint()

        log_to_stderr()
        self.__df = pd.read_csv(self.__source_dataframe, nrows=1000 if self.__debug else None)
        self.__pivot()

        self._log("Source dataframe size: {}".format(self.__df.shape[0]))
        self.__pbar = tqdm(total=self.__df.shape[0], smoothing=.1)
        self.__pool = Pool(self.__n_jobs)

        for i, row in self.__df.iterrows():

            new_row = row.to_dict()

            self.__pool.apply_async(process_helper,
                                    args=(i, row['id'], new_row),
                                    callback=self.__update,
                                    error_callback=self.__error)

        self.__pool.close()
        self.__pool.join()
        self.__pbar.close()

        self._save()

    def __error(self, wrapper):
        """Multiprocess error callback"""
        # return get_logger().error(msg, *args)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, file=sys.stdout)
        print(wrapper)

    def process(self, i, row_id, new_row):
        """DICOM - numpy konvertáló

            - Ha a folyamat félbe szakad, akkor a korábban már feldolgozott képeket nem futtatja újra
              (Nem működik 100%-ban)
            - Beolvassa a DICOM forrás file-t
            - A DICOM metaadatokban tárolt értékek (RescaleIntercept, RescaleSlope) konvertálja az adatokat
              Hounsfield értekekre.
            - Az eltérő PixelSpacing értékek miatt egységesíti az adatokat (resample)
            - Maszkolja és kivágja a képen a koponya körüli részt (minden mást levág)
            - Négyzet alakúra igazítja a képet
            - Módosítja a méretet a beállított cél értékre (resize / pad)
            - Normalizál
            - Zero-centering (eltolja az értékeket, hogy az átlag 0 legyen)
        """
        dicom_file = self.__input_folder + '/stage_{}_{}_images/{}.dcm'.format(
            self.__stage, self.__dataset, row_id)

        if not os.path.isfile(dicom_file) or row_id in self.__processed:
            # Missing file (already processed and deleted?!)
            self._log("{} already processed, skip.".format(row_id))
            return None

        dicom = pydicom.read_file(dicom_file)

        if self.__save_dicom_metadata:
            new_row = self.__extract_dicom_metadata(row_id, new_row, dicom)

        # CT (HU values) as numpy array
        ct = self.__get_pixels_hu(dicom.pixel_array, intercept=dicom.RescaleIntercept, slope=dicom.RescaleSlope)

        if self.__save_dicom_metadata:
            # We've already converted to HU values
            new_row['RescaleIntercept'] = 0
            new_row['RescaleSlope'] = 1

        if self.__target_pixel_spacing[0] != 0:
            ct, spacing = self.__resample(ct, dicom.PixelSpacing, self.__target_pixel_spacing)

            if self.__save_dicom_metadata:
                new_row['PixelSpacing_x'] = self.__target_pixel_spacing[0]
                new_row['PixelSpacing_y'] = self.__target_pixel_spacing[1]

        if self.__mask_head:
            # Remove everything outside of the head
            # cv2.imwrite(self.__output_folder_image + '/' + row_id + '_before_mask.png', ct)

            mask, ct = self.__mask(ct)

            if self.__crop_head:
                # Crop based on the size of the head
                # cv2.imwrite(self.__output_folder_image + '/' + row_id + '_before_crop.png', ct)

                ct = self.__crop(ct, mask)

        if self.__target_size is not None:
            # square, pad, resize
            ct = self.__resize(ct)

        if self.__channels:
            # Convert dicom to rgb
            ct = self.__convert_to_rgb(ct)

        if self.__normalize[0] != 0:
            # Normalize
            ct = self.__do_normalize(ct)

        if self.__save_dicom_metadata:
            new_row['Rows'] = ct.shape[0]
            new_row['Columns'] = ct.shape[1]

            if self.__save_pixel_stats:
                doctor = self.__window_image(ct, new_row['WindowCenter'], new_row['WindowWidth'])
                custom = self.__window_image(ct, 40, 80)

                # new_row['px_raw_min'] = ct.min()
                # new_row['px_raw_max'] = ct.min()
                # new_row['px_raw_mean'] = ct.mean()
                # new_row['px_raw_diff'] = ct.max() - ct.min()

                new_row['px_custom_min'] = custom.min()
                new_row['px_custom_max'] = custom.min()
                new_row['px_custom_mean'] = custom.mean()
                new_row['px_custom_diff'] = custom.max() - custom.min()

                # new_row['px_doctor_min'] = doctor.min()
                # new_row['px_doctor_max'] = doctor.min()
                # new_row['px_doctor_mean'] = doctor.mean()
                # new_row['px_doctor_diff'] = doctor.max() - doctor.min()

                """ multichannel
                for i in range(3):

                    # new_row['PixelCount_{}'.format(i)] = np.size(ct)
                    new_row['PixelSum_{}'.format(i)] = ct[:, :, i].sum()
                    new_row['PixelMean_{}'.format(i)] = ct[:, :, i].mean()
                    new_row['PixelStd_{}'.format(i)] = ct[:, :, i].std()
                    new_row['PixelMin_{}'.format(i)] = ct[:, :, i].min()
                    new_row['PixelMax_{}'.format(i)] = ct[:, :, i].max()
                """

        # Add new row
        # self.__data.append(new_row)

        # Save output.
        if self.__output_type == 'npz':
            # cv2.imwrite(self.__output_folder_image + '/' + row_id + '.png', ct)
            np.savez_compressed(self.__output_folder_image + '/' + row_id + '.npz', ct)
        elif self.__output_type == 'png':
            ct = ct.astype(np.uint8)
            cv2.imwrite(self.__output_folder_image + '/' + row_id + '.png', ct)

        if self.__remove_dicom_files:
            os.remove(dicom_file)

        return new_row

    def __window_image(self, image, window_center, window_width):
        img = image.copy()
        # img = img.astype(np.float32)
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2

        img[img < img_min] = img_min
        img[img > img_max] = img_max

        # img = (img - np.min(img)) / (np.max(img) - np.min(img))

        return img

    def __convert_to_rgb(self, ct):

        image = ct.copy()

        brain_img = self.__window_image(image, 40, 80)
        subdural_img = self.__window_image(image, 80, 200)
        bone_img = self.__window_image(image, 40, 380)

        brain_img = (brain_img - 0) / 80
        subdural_img = (subdural_img - (-20)) / 200
        bone_img = (bone_img - (-150)) / 380

        bsb_img = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.float32)
        bsb_img[:, :, 0] = brain_img  # - brain_img.mean()
        bsb_img[:, :, 1] = subdural_img  # - subdural_img.mean()
        bsb_img[:, :, 2] = bone_img  # - bone_img.mean()

        return (bsb_img * 255).astype(np.uint8)

    def __update(self, res):
        """tqdm állapot frissítés"""
        if res is not None:
            self.__data.append(res)

        self.__pbar.update()

    def _save(self):
        """Menti a a meta-, pixel-adatokkal együtt a dataframe-et"""
        df = pd.DataFrame(self.__data)
        
        # Remove useless images (less than 60 'brain' pixel value)
        df = df[df['px_custom_diff'] > 60]
        
        df = df[['id', 'any', 'cls_1', 'cls_2', 'cls_3', 'cls_4', 'cls_5', 'PatientID', 'WindowCenter',
                 'WindowWidth']]

        df['sample_type'] = self.__dataset

        df = df.sort_values('id')

        # print(df.shape)
        df.to_csv(self.__output_folder + '/' + self.__df_name, index=False)

    def _log(self, msg):
        # TODO: do your logging here.
        print(msg)

    def __pivot(self):
        self._log("Preparing dataframe...")
        self.__df[['_id_', 'id', 'diagnosis']] = self.__df['ID'].str.split('_', expand=True)
        self.__df = self.__df[['id', 'diagnosis', 'Label']]
        self.__df.drop_duplicates(inplace=True)
        self.__df = self.__df.pivot(index='id', columns='diagnosis', values='Label').reset_index()
        self.__df['id'] = 'ID_' + self.__df['id']
        # self.__df['sample_type'] = 'train'
        self.__df = self.__df.fillna(0)

        self.__df = self.__df.astype({'any': 'uint8', 'epidural': 'uint8', 'intraparenchymal': 'uint8',
                                      'intraventricular': 'uint8', 'subarachnoid': 'uint8', 'subdural': 'uint8'})

        self.__df = self.__df.rename(columns={'epidural': 'cls_1', 'intraparenchymal': 'cls_2',
                                              'intraventricular': 'cls_3', 'subarachnoid': 'cls_4',
                                              'subdural': 'cls_5'})
        self._log("Dataframe is ready.")

    def __do_normalize(self, ct):
        """Normalize, zero centering"""
        ct = (ct - self.__normalize[0]) / (self.__normalize[1] - self.__normalize[0])
        ct[ct > 1] = 1.
        ct[ct < 0] = 0.

        ct = ct - self.__zero_center

        return ct

    def __resize(self, ct):
        """Módosítja a képé méreteit

            - Négyzetté igazítja ha kell
            - Ha a kép a cél méretnél nagyobb, akkor kicsinyíti (cv2.resize)
            - Ha a kép a cél méretnél kisebb, akkor keretezi (cv2.copyMakeBorder) - BUG?
        """
        size = self.__target_size
        height, width = ct.shape
        bg = ct[1, 1]

        if height == width:
            square_ct = ct
        else:
            sq_size = max(width, height)
            square_ct = np.ones((sq_size, sq_size)) * bg

            if height > width:
                x1 = sq_size // 2 - width // 2
                square_ct[0:height, x1:x1+width] = ct

            else:
                y1 = sq_size // 2 - height // 2
                square_ct[y1:y1+height, 0:width] = ct

        if square_ct.shape[0] > size:
            # down -> cv2 resize
            square_ct = cv2.resize(square_ct, (size, size))
        elif square_ct.shape[0] < size:
            # pad
            sq_size = square_ct.shape[0]
            border = (size - sq_size) // 2
            square_ct = cv2.copyMakeBorder(square_ct, border, border, border, border, dst=None,
                                           borderType=cv2.BORDER_CONSTANT, value=int(bg))

        if square_ct.shape[0] != self.__target_size:
            square_ct = cv2.resize(square_ct, (self.__target_size, self.__target_size))

        return square_ct

    def __mask(self, ct):
        """"Maszkolja a koponyát"""
        # TODO: Use config instead of fix: -720
        binary_image = np.array(ct > -720, dtype=np.int8)

        if binary_image.sum() < 1000:
            return np.ones(binary_image.shape, dtype=np.int8), ct

        bg = binary_image[0, 0]
        labels = measure.label(binary_image, background=bg)

        vals, counts = np.unique(labels, return_counts=True)
        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            l_max = vals[np.argmax(counts)]
        else:
            return np.ones(binary_image.shape, dtype=np.int8), ct

        binary_image[labels != l_max] = 0
        binary_image[labels == l_max] = 1

        ct[binary_image == 0] = 0

        return binary_image, ct

    def __crop(self, ct, mask):
        """Kivágja a maszkolt részt (átméretezi a képet, hogy csak a maszkolt rész legyen rajta)"""
        mask = mask == 0

        # Find the bounding box of those pixels
        coords = np.array(np.nonzero(~mask))
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)

        x1 = top_left[0]
        y1 = top_left[1]
        x2 = bottom_right[0]
        y2 = bottom_right[1]

        return ct[x1:x2, y1:y2]

    def __extract_dicom_metadata(self, row_id, new_row, dicom):
        """"Kiolvassa és tárolja a DICOM metaadatokat"""

        columns = ['SOPInstanceUID', 'Modality', 'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'StudyID',
                   'ImagePositionPatient_x', 'ImagePositionPatient_y', 'ImagePositionPatient_z',
                   'ImageOrientationPatient_0', 'ImageOrientationPatient_1', 'ImageOrientationPatient_2',
                   'ImageOrientationPatient_3', 'ImageOrientationPatient_4', 'ImageOrientationPatient_5',
                   'SamplesPerPixel', 'PhotometricInterpretation', 'Rows', 'Columns', 'PixelSpacing_x',
                   'PixelSpacing_y',
                   'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation', 'WindowCenter', 'WindowWidth',
                   'RescaleIntercept', 'RescaleSlope']

        for element in dicom:

            try:
                if element.keyword == 'ImagePositionPatient':
                    new_row[element.keyword + '_x'] = element.value[0]
                    new_row[element.keyword + '_y'] = element.value[1]
                    new_row[element.keyword + '_z'] = element.value[2]

                elif element.keyword == 'ImageOrientationPatient':
                    new_row[element.keyword + '_0'] = element.value[0]
                    new_row[element.keyword + '_1'] = element.value[1]
                    new_row[element.keyword + '_2'] = element.value[2]
                    new_row[element.keyword + '_3'] = element.value[3]
                    new_row[element.keyword + '_4'] = element.value[4]
                    new_row[element.keyword + '_5'] = element.value[5]

                elif element.keyword == 'PixelSpacing':
                    new_row[element.keyword + '_x'] = element.value[0]
                    new_row[element.keyword + '_y'] = element.value[1]

                elif element.keyword == 'WindowCenter' and isinstance(element.value, pydicom.multival.MultiValue):
                    new_row[element.keyword] = element.value[0]

                elif element.keyword == 'WindowWidth' and isinstance(element.value, pydicom.multival.MultiValue):
                    new_row[element.keyword] = element.value[0]

                elif element.keyword == 'PixelData':
                    continue
                else:
                    if element.keyword not in columns:
                        continue

                    new_row[element.keyword] = element.value

            except:
                self._log("Error processing dicom metadata: {}".format(row_id))
                return None

        return new_row

    def __get_pixels_hu(self, image, intercept=0, slope=1):
        """HU értekekre konvertálja a dicom pixel adatokat"""
        image = image.astype(np.int16)
        image[image == -2000] = 0

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)

        return np.array(image, dtype=np.int16)

    def __resample(self, image, pixel_spacing, new_spacing=(1, 1)):
        """PixelSpacing normalizálás"""
        spacing = np.array(pixel_spacing)

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

        return image, new_spacing

    def __load_checkpoint(self):

        if os.path.isfile(self.__output_folder + '/' + self.__df_name):
            self._log("Loading checkpoint...")
            df = pd.read_csv(self.__output_folder + '/' + self.__df_name)
            self.__processed = df['id'].values
            self.__data = df.to_dict(orient='records')
            self._log("{} records loaded sucessfully".format(df.shape[0]))


if __name__ == '__main__':

    # PREPROCESSED C (Appian)
    rsna_converter = RSNADatasetConverter(
        source_dataframe='./input2/stage_1_sample_submission.csv',
        input_folder='./input2',
        output_folder='./input2/preprocessed_c_dicom',
        stage=1,
        dataset='test',
        save_dicom_metadata=True,
        save_pixel_stats=True,
        target_pixel_spacing=(0, 0),
        mask_head=False,
        crop_head=False,
        target_size=None,
        normalize=(0, 0),
        channels=False,
        zero_center=False,
        ouput_type='no-op',

        n_jobs=10,
        remove_dicom_files=False,
        debug=False
    )

    """
    # PREPROCESSED B
    rsna_converter = RSNADatasetConverter(
        source_dataframe='./input2/stage_1_sample_submission.csv',
        input_folder='./input2',
        output_folder='./input2/preprocessed_png_224',
        stage=1,
        dataset='test',
        save_dicom_metadata=True,
        save_pixel_stats=True,
        target_pixel_spacing=(0, 0),
        mask_head=True,
        crop_head=True,
        target_size=224,
        normalize=(0, 0),
        channels=True,
        zero_center=False,
        ouput_type='png',

        n_jobs=10,
        remove_dicom_files=False,
        debug=False
    )
    """

    rsna_converter.convert()
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback, file=sys.stdout)
