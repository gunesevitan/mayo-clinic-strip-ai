import sys
import logging
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

sys.path.append('..')
import settings


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train.csv')
    df_test = pd.read_csv(settings.DATA / 'test.csv')
    df_other = pd.read_csv(settings.DATA / 'other.csv')

    train_images_filenames = glob(f'{str(settings.DATA / "train_compressed")}/*')
    test_images_filenames = glob(f'{str(settings.DATA / "test_compressed")}/*')
    other_images_filenames = glob(f'{str(settings.DATA / "other_compressed")}/*')

    logging.info(f'Training Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')
    logging.info(f'Other Shape: {df_other.shape} - Memory Usage: {df_other.memory_usage().sum() / 1024 ** 2:.2f} MB')

    for image_filename in tqdm(train_images_filenames):

        image_id = image_filename.split('/')[-1].split('.')[0]

        # Extract metadata from image
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        df_train.loc[df_train['image_id'] == image_id, 'image_r_mean'] = np.mean(image[:, :, 0])
        df_train.loc[df_train['image_id'] == image_id, 'image_r_std'] = np.std(image[:, :, 0])
        df_train.loc[df_train['image_id'] == image_id, 'image_g_mean'] = np.mean(image[:, :, 1])
        df_train.loc[df_train['image_id'] == image_id, 'image_g_std'] = np.std(image[:, :, 1])
        df_train.loc[df_train['image_id'] == image_id, 'image_b_mean'] = np.mean(image[:, :, 2])
        df_train.loc[df_train['image_id'] == image_id, 'image_b_std'] = np.std(image[:, :, 2])

        image_shape = image.shape
        df_train.loc[df_train['image_id'] == image_id, 'image_height'] = image_shape[0]
        df_train.loc[df_train['image_id'] == image_id, 'image_width'] = image_shape[1]
        df_train.loc[df_train['image_id'] == image_id, 'image_filename'] = image_filename

    df_train.to_csv(settings.DATA / 'train_metadata.csv', index=False)
    logging.info(f'Saved train_metadata.csv to {settings.DATA}')
    logging.info(f'Training Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    for image_filename in tqdm(test_images_filenames):

        image_id = image_filename.split('/')[-1].split('.')[0]

        # Extract metadata from image
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        df_test.loc[df_test['image_id'] == image_id, 'image_r_mean'] = np.mean(image[:, :, 0])
        df_test.loc[df_test['image_id'] == image_id, 'image_r_std'] = np.std(image[:, :, 0])
        df_test.loc[df_test['image_id'] == image_id, 'image_g_mean'] = np.mean(image[:, :, 1])
        df_test.loc[df_test['image_id'] == image_id, 'image_g_std'] = np.std(image[:, :, 1])
        df_test.loc[df_test['image_id'] == image_id, 'image_b_mean'] = np.mean(image[:, :, 2])
        df_test.loc[df_test['image_id'] == image_id, 'image_b_std'] = np.std(image[:, :, 2])

        image_shape = image.shape
        df_test.loc[df_test['image_id'] == image_id, 'image_height'] = image_shape[0]
        df_test.loc[df_test['image_id'] == image_id, 'image_width'] = image_shape[1]
        df_test.loc[df_test['image_id'] == image_id, 'image_filename'] = image_filename

    df_test.to_csv(settings.DATA / 'test_metadata.csv', index=False)
    logging.info(f'Saved test_metadata.csv to {settings.DATA}')
    logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    for image_filename in tqdm(other_images_filenames):

        image_id = image_filename.split('/')[-1].split('.')[0]

        # Extract metadata from image
        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        df_other.loc[df_other['image_id'] == image_id, 'image_r_mean'] = np.mean(image[:, :, 0])
        df_other.loc[df_other['image_id'] == image_id, 'image_r_std'] = np.std(image[:, :, 0])
        df_other.loc[df_other['image_id'] == image_id, 'image_g_mean'] = np.mean(image[:, :, 1])
        df_other.loc[df_other['image_id'] == image_id, 'image_g_std'] = np.std(image[:, :, 1])
        df_other.loc[df_other['image_id'] == image_id, 'image_b_mean'] = np.mean(image[:, :, 2])
        df_other.loc[df_other['image_id'] == image_id, 'image_b_std'] = np.std(image[:, :, 2])

        image_shape = image.shape
        df_other.loc[df_other['image_id'] == image_id, 'image_height'] = image_shape[0]
        df_other.loc[df_other['image_id'] == image_id, 'image_width'] = image_shape[1]
        df_other.loc[df_other['image_id'] == image_id, 'image_filename'] = image_filename

    df_other.to_csv(settings.DATA / 'other_metadata.csv', index=False)
    logging.info(f'Saved other_metadata.csv to {settings.DATA}')
    logging.info(f'Other Shape: {df_other.shape} - Memory Usage: {df_other.memory_usage().sum() / 1024 ** 2:.2f} MB')
