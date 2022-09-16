import logging
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import settings
import image_utils


def visualize_categorical_feature_distribution(df, categorical_feature, path=None):

    """
    Visualize distribution of given categorical feature in given dataframe

    Parameters
    ----------
    df (pandas.DataFrame of shape (n_rows, n_columns)): Dataframe with given categorical feature column
    categorical_feature (str): Name of the categorical feature column
    path (path-like str or None): Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, df[categorical_feature].value_counts().shape[0] + 6), dpi=100)
    sns.barplot(
        x=df[categorical_feature].value_counts().values,
        y=df[categorical_feature].value_counts().index,
        color='tab:blue',
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticklabels([
        f'{x} ({value_count:,})' for value_count, x in zip(
            df[categorical_feature].value_counts().values,
            df[categorical_feature].value_counts().index
        )
    ])
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(f'Value Counts {categorical_feature}', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_image(image, metadata, path=None):

    """
    Visualize image along with its annotations

    Parameters
    ----------
    image (path-like str or numpy.ndarray of shape (height, width, 3)): Image path relative to root/data or image array
    metadata (dict): Dictionary of metadata used in the visualization title
    path (path-like str or None): Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    if isinstance(image, pathlib.Path) or isinstance(image, str):
        # Read image from the given path
        image_path = image
        image = cv2.imread(str(settings.DATA / image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Raise TypeError if image argument is not an array-like object or a path-like string
        raise TypeError('Image is not an array or path.')

    fig, ax = plt.subplots(figsize=(16, 16))

    ax.imshow(image)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(
        f'''
        Image ID: {metadata["image_id"]} - Center ID: {metadata["center_id"]} - Patient ID: {metadata["patient_id"]} - Image Number {metadata["image_num"]}
        Label: {metadata["label"]}
        Image Shape: {int(metadata["image_height"])}x{int(metadata["image_width"])}
        ''',
        size=20,
        pad=15
    )

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_transforms(image, transforms, path=None):

    """
    Visualize image along with its annotations and predictions

    Parameters
    ----------
    image (numpy.ndarray of shape (height, width, 3)): Image array
    transforms (albumentations.Compose): Transforms to apply on image
    path (path-like str or None): Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, axes = plt.subplots(figsize=(32, 20), ncols=2)
    axes[0].imshow(image)

    # Apply transforms to image
    transformed = transforms(image=image)
    transformed_image = transformed['image']
    axes[1].imshow(transformed_image)

    for i in range(2):
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=15, pad=10)
        axes[i].tick_params(axis='y', labelsize=15, pad=10)

    axes[0].set_title(f'{image.shape} - Mean: {image.mean():.2f} - Std: {image.std():.2f}\nMin: {image.min():.2f} - Max: {image.max():.2f}', size=20, pad=15)
    axes[1].set_title(f'{transformed_image.shape} - Mean: {transformed_image.mean():.2f} - Std: {transformed_image.std():.2f}\nMin: {transformed_image.min():.2f} - Max: {transformed_image.max():.2f}', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_learning_curve(training_losses, validation_losses, path=None):

    """
    Visualize learning curves of the models

    Parameters
    ----------
    training_losses (array-like of shape (n_epochs or n_steps)): Array of training losses computed after every epoch or step
    validation_losses (array-like of shape (n_epochs or n_steps)): Array of validation losses computed after every epoch or step
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(32, 8), dpi=100)

    sns.lineplot(
        x=np.arange(1, len(training_losses) + 1),
        y=training_losses,
        ax=ax,
        label='train_loss'
    )

    if validation_losses is not None:
        sns.lineplot(
            x=np.arange(1, len(validation_losses) + 1),
            y=validation_losses,
            ax=ax,
            label='val_loss'
        )

    ax.set_xlabel('Epochs/Steps', size=15, labelpad=12.5)
    ax.set_ylabel('Loss', size=15, labelpad=12.5)
    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.legend(prop={'size': 18})
    ax.set_title('Learning Curve', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_scores(df_scores, path=None):

    """
    Visualize metric scores of multiple models with error bars

    Parameters
    ----------
    df_scores (pandas.DataFrame of shape (n_folds, 6)): DataFrame of scores
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    # Create mean and std of scores for error bars
    df_scores = df_scores.T
    column_names = df_scores.columns.to_list()
    df_scores['mean'] = df_scores[column_names].mean(axis=1)
    df_scores['std'] = df_scores[column_names].std(axis=1)

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.barh(
        y=np.arange(df_scores.shape[0]),
        width=df_scores['mean'],
        xerr=df_scores['std'],
        align='center',
        ecolor='black',
        capsize=10
    )
    ax.set_yticks(np.arange(df_scores.shape[0]))
    ax.set_yticklabels([
        f'{metric}\n{mean:.4f} (±{std:.4f})' for metric, mean, std in zip(
            df_scores.index,
            df_scores['mean'].values,
            df_scores['std'].values
        )
    ])
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title('Metric Scores', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train_metadata.csv')
    df_other = pd.read_csv(settings.DATA / 'other_metadata.csv')
    df_other['center_id'] = np.nan

    VISUALIZE_IMAGES = False
    VISUALIZE_CATEGORICAL_FEATURES = True

    if VISUALIZE_IMAGES:

        train_image_visualizations_directory = settings.EDA / 'train_compressed_images'
        train_image_visualizations_directory.mkdir(parents=True, exist_ok=True)

        for idx, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):

            image = cv2.imread(row['image_filename'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image_utils.resize_with_aspect_ratio(image, longest_edge=2000)

            visualize_image(
                image=image,
                metadata=row.to_dict(),
                path=train_image_visualizations_directory / f'{row["image_id"]}.png'
            )

        logging.info(f'Saved train image visualizations to {train_image_visualizations_directory}')

        other_image_visualizations_directory = settings.EDA / 'other_compressed_images'
        other_image_visualizations_directory.mkdir(parents=True, exist_ok=True)

        for idx, row in tqdm(df_other.iterrows(), total=df_other.shape[0]):

            image = cv2.imread(row['image_filename'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image_utils.resize_with_aspect_ratio(image, longest_edge=2000)

            visualize_image(
                image=image,
                metadata=row.to_dict(),
                path=other_image_visualizations_directory / f'{row["image_id"]}.png'
            )

        logging.info(f'Saved other image visualizations to {other_image_visualizations_directory}')

    if VISUALIZE_CATEGORICAL_FEATURES:

        train_categorical_features = ['label', 'center_id']
        for categorical_feature in train_categorical_features:
            visualize_categorical_feature_distribution(
                df=df_train,
                categorical_feature=categorical_feature,
                path=settings.EDA / f'train_{categorical_feature}_distribution.png'
            )
            logging.info(f'Saved train categorical feature {categorical_feature} distribution visualizations to {settings.EDA}')

        other_categorical_features = ['label', 'other_specified']
        for categorical_feature in other_categorical_features:
            visualize_categorical_feature_distribution(
                df=df_other,
                categorical_feature=categorical_feature,
                path=settings.EDA / f'other_{categorical_feature}_distribution.png'
            )
            logging.info(f'Saved other categorical feature {categorical_feature} distribution visualizations to {settings.EDA}')
