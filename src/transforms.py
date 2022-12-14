import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import ImageOnlyTransform
import staintools


class Scale(ImageOnlyTransform):

    def apply(self, image, **kwargs):

        """
        Scale pixel values between 0 and 1

        Parameters
        ----------
        image (numpy.ndarray of shape (height, width, channel)): Image array

        Returns
        -------
        image (numpy.ndarray of shape (height, width, channel)): Image array divided by max 8-bit integer
        """

        image = np.float32(image) / 255.

        return image


class StandardizeLuminosity(ImageOnlyTransform):

    def apply(self, image, **kwargs):

        """
        Normalize luminosity (white areas) in whole-slide images

        Parameters
        ----------
        image (numpy.ndarray of shape (height, width, channel)): Image array

        Returns
        -------
        image (numpy.ndarray of shape (height, width, channel)): Image array with standardized luminosity
        """

        image = staintools.LuminosityStandardizer.standardize(image)

        return image


def get_classification_transforms(**transform_parameters):

    """
    Get transforms for classification dataset

    Parameters
    ----------
    transform_parameters (dict): Dictionary of transform parameters

    Returns
    -------
    transforms (dict): Transforms for training, validation and test sets
    """

    train_transforms = A.Compose([
        A.Resize(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            interpolation=cv2.INTER_NEAREST,
            always_apply=True
        ),
        StandardizeLuminosity(p=transform_parameters['standardize_luminosity_probability']),
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.RandomRotate90(p=transform_parameters['random_rotate_90_probability']),
        A.HueSaturationValue(
            hue_shift_limit=transform_parameters['hue_shift_limit'],
            sat_shift_limit=transform_parameters['saturation_shift_limit'],
            val_shift_limit=transform_parameters['value_shift_limit'],
            p=transform_parameters['hue_saturation_value_probability']
        ),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            max_pixel_value=transform_parameters['normalize_max_pixel_value'],
            always_apply=True
        ),
        A.OneOf([
            A.CoarseDropout(
                max_holes=transform_parameters['coarse_dropout_max_holes'],
                max_height=transform_parameters['coarse_dropout_max_height'],
                max_width=transform_parameters['coarse_dropout_max_width'],
                min_holes=transform_parameters['coarse_dropout_min_holes'],
                min_height=transform_parameters['coarse_dropout_min_height'],
                min_width=transform_parameters['coarse_dropout_min_width'],
                fill_value=transform_parameters['coarse_dropout_fill_value'],
                p=transform_parameters['coarse_dropout_probability']
            ),
            A.PixelDropout(
                dropout_prob=transform_parameters['pixel_dropout_dropout_probability'],
                per_channel=transform_parameters['pixel_dropout_per_channel'],
                drop_value=transform_parameters['pixel_dropout_drop_value'],
                p=transform_parameters['pixel_dropout_probability']
            )
        ], p=transform_parameters['dropout_probability']),
        ToTensorV2(always_apply=True)
    ])

    val_transforms = A.Compose([
        A.Resize(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            interpolation=cv2.INTER_NEAREST,
            always_apply=True
        ),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            max_pixel_value=transform_parameters['normalize_max_pixel_value'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    test_transforms = A.Compose([
        A.Resize(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            interpolation=cv2.INTER_NEAREST,
            always_apply=True
        ),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            max_pixel_value=transform_parameters['normalize_max_pixel_value'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    transforms = {'train': train_transforms, 'val': val_transforms, 'test': test_transforms}
    return transforms
