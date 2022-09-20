import numpy as np
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
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            max_pixel_value=transform_parameters['normalize_max_pixel_value'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    val_transforms = A.Compose([
        A.Normalize(
            mean=transform_parameters['normalize_mean'],
            std=transform_parameters['normalize_std'],
            max_pixel_value=transform_parameters['normalize_max_pixel_value'],
            always_apply=True
        ),
        ToTensorV2(always_apply=True)
    ])

    test_transforms = A.Compose([
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