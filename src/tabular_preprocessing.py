import numpy as np


def _get_folds(df, df_folds):

    """
    Merge training set with pre-computed folds

    Parameters
    ----------
    df (pandas.DataFrame of shape (n_rows, n_columns)): Training dataframe
    df_folds (pandas.DataFrame of shape (n_rows, n_folds + 1)): Folds dataframe

    Returns
    -------
    df (pandas.DataFrame of shape (n_rows, n_columns + n_folds)): Training dataframe merged with folds dataframe
    """

    df = df.merge(df_folds, on='image_id', how='left')

    return df


def _encode_target(df):

    """
    Numerically encode categorical target column

    Parameters
    ----------
    df (pandas.DataFrame of shape (n_rows, n_columns)): Training dataframe with label column

    Returns
    -------
    df (pandas.DataFrame of shape (n_rows, n_columns + 2)): Training dataframe with binary and multiclass encoded label columns
    """

    df['binary_encoded_label'] = df['label'].map({'CE': 0, 'LAA': 1}).astype(np.uint8)
    df['multiclass_encoded_label'] = df['label'].map({'CE': 1, 'LAA': 2}).astype(np.uint8)

    return df


def process_datasets(df_train, df_test, df_folds):

    """
    Preprocess training and test sets

    Parameters
    ----------
    df_train (pandas.DataFrame of shape (n_rows, n_columns)): Training dataframe
    df_test (pandas.DataFrame of shape (n_rows, n_columns)): Test dataframe
    df_folds (pandas.DataFrame of shape (n_rows, n_folds)): Folds dataframe

    Returns
    -------
    df_train (pandas.DataFrame of shape (n_rows, n_columns)): Processed training dataframe
    df_test (pandas.DataFrame of shape (n_rows, n_columns)): Processed test dataframe
    """

    df_train = _get_folds(df=df_train, df_folds=df_folds)
    df_train = _encode_target(df=df_train)

    return df_train, df_test
