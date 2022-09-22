import sys
import os
import logging
import pathlib
import numpy as np
import pandas as pd
import cv2
import tifffile
import pyvips

sys.path.append('..')
import settings


MAX_SIZE = 20000
JPEG_QUALITY = 100
DTYPE_MAPPING = {
   'uchar': np.uint8,
   'char': np.int8,
   'ushort': np.uint16,
   'short': np.int16,
   'uint': np.uint32,
   'int': np.int32,
   'float': np.float32,
   'double': np.float64,
   'complex': np.complex64,
   'dpcomplex': np.complex128,
}


def vips_to_numpy(image_thumbnail):

    return np.ndarray(
        buffer=image_thumbnail.write_to_memory(),
        dtype=DTYPE_MAPPING[image_thumbnail.format],
        shape=[image_thumbnail.height, image_thumbnail.width, image_thumbnail.bands]
    )


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train.csv')
    df_test = pd.read_csv(settings.DATA / 'test.csv')
    df_other = pd.read_csv(settings.DATA / 'other.csv')

    train_images = settings.DATA / 'train'
    test_images = settings.DATA / 'test'
    other_images = settings.DATA / 'other'

    logging.info(f'Training Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')
    logging.info(f'Other Shape: {df_other.shape} - Memory Usage: {df_other.memory_usage().sum() / 1024 ** 2:.2f} MB')

    pathlib.Path(str(settings.DATA / 'train_compressed')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(str(settings.DATA / 'test_compressed')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(str(settings.DATA / 'other_compressed')).mkdir(parents=True, exist_ok=True)

    for idx, row in df_train.iterrows():

        image_thumbnail = pyvips.Image.thumbnail(f'{train_images}/{row["image_id"]}.tif', MAX_SIZE)
        image = vips_to_numpy(image_thumbnail=image_thumbnail)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_image_megabytes = image.nbytes / (1024 ** 2)
        cv2.imwrite(str(settings.DATA / 'train_compressed' / f'{row["image_id"]}.jpg'), image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        compressed_image_megabytes = os.path.getsize(str(settings.DATA / 'train_compressed' / f'{row["image_id"]}.jpg')) / (1024 ** 2)

        logging.info(f'Image {row["image_id"]} Raw Size: {raw_image_megabytes:.4f} Compressed Size: {compressed_image_megabytes:.4f}')

    for idx, row in df_test.iterrows():

        image_thumbnail = pyvips.Image.thumbnail(f'{test_images}/{row["image_id"]}.tif', MAX_SIZE)
        image = vips_to_numpy(image_thumbnail=image_thumbnail)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_image_megabytes = image.nbytes / (1024 ** 2)
        cv2.imwrite(str(settings.DATA / 'test_compressed' / f'{row["image_id"]}.jpg'), image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        compressed_image_megabytes = os.path.getsize(str(settings.DATA / 'test_compressed' / f'{row["image_id"]}.jpg')) / (1024 ** 2)

        logging.info(f'Image {row["image_id"]} Raw Size: {raw_image_megabytes:.4f} Compressed Size: {compressed_image_megabytes:.4f}')

    for idx, row in df_other.iterrows():

        image_thumbnail = pyvips.Image.thumbnail(f'{test_images}/{row["image_id"]}.tif', MAX_SIZE)
        image = vips_to_numpy(image_thumbnail=image_thumbnail)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_image_megabytes = image.nbytes / (1024 ** 2)
        cv2.imwrite(str(settings.DATA / 'test_compressed' / f'{row["image_id"]}.jpg'), image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        compressed_image_megabytes = os.path.getsize(str(settings.DATA / 'other_compressed' / f'{row["image_id"]}.jpg')) / (1024 ** 2)

        logging.info(f'Image {row["image_id"]} Raw Size: {raw_image_megabytes:.4f} Compressed Size: {compressed_image_megabytes:.4f}')
