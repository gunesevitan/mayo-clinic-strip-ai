import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import torch


def soft_predictions_to_labels(x, threshold):

    """
    Convert soft predictions into hard labels in given array

    Parameters
    ----------
    x (array-like of any shape): Soft predictions array
    threshold (float): Threshold for converting soft predictions into hard labels (0 <= threshold <= 1)

    Returns
    -------
    x (array-like of any shape): Hard labels array
    """

    if isinstance(x, torch.Tensor):
        x = x.numpy()
    else:
        x = np.array(x)

    x = np.uint8(x >= threshold)

    return x


def binary_weighted_log_loss(y_true, y_pred):

    """
    Calculate positive, negative and weighted log loss on predictions and ground-truth

    Parameters
    ----------
    y_true (array-like of shape (n_samples)): Ground-truth
    y_pred (array-like of shape (n_samples)): Predictions

    Returns
    -------
    weighted_log_loss (float): Weighted log loss score calculated on predictions and ground-truth
    """

    log_loss_positive = log_loss(y_true, y_pred)
    log_loss_negative = log_loss(y_true, 1 - y_pred)
    log_loss_weighted = 0.5 * log_loss_positive + 0.5 * log_loss_negative

    return log_loss_positive, log_loss_negative, log_loss_weighted


def binary_classification_scores(y_true, y_pred, threshold):

    """
    Calculate binary classification metrics on predictions and ground-truth

    Parameters
    ----------
    y_true (array-like of shape (n_samples)): Ground-truth
    y_pred (array-like of shape (n_samples)): Predictions
    threshold (float): Threshold for converting soft predictions into hard labels (0 <= threshold <= 1)

    Returns
    -------
    scores (dict): Dictionary of scores
    """

    accuracy = accuracy_score(y_true, soft_predictions_to_labels(y_pred, threshold=threshold))
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = 0.5
    log_loss_positive, log_loss_negative, log_loss_weighted = binary_weighted_log_loss(y_true, y_pred)

    scores = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'log_loss_positive': log_loss_positive,
        'log_loss_negative': log_loss_negative,
        'log_loss_weighted': log_loss_weighted
    }

    return scores


def multiclass_weighted_log_loss(y_true, y_pred):

    """
    Calculate positive, negative and weighted log loss on predictions and ground-truth

    Parameters
    ----------
    y_true (array-like of shape (n_samples)): Ground-truth
    y_pred (array-like of shape (n_samples)): Predictions

    Returns
    -------
    weighted_log_loss (float): Weighted log loss score calculated on predictions and ground-truth
    """

    log_loss_positive = log_loss(y_true, y_pred[:, 1])
    log_loss_negative = log_loss(y_true, y_pred[:, 0])
    log_loss_weighted = 0.5 * log_loss_positive + 0.5 * log_loss_negative

    return log_loss_positive, log_loss_negative, log_loss_weighted


def multiclass_classification_scores(y_true, y_pred):

    """
    Calculate multiclass classification metrics on predictions and ground-truth

    Parameters
    ----------
    y_true (array-like of shape (n_samples)): Ground-truth
    y_pred (array-like of shape (n_samples)): Predictions

    Returns
    -------
    scores (dict): Dictionary of scores
    """

    accuracy = accuracy_score(y_true, np.argmax(y_pred, axis=1))
    try:
        roc_auc = np.mean([roc_auc_score(y_true, y_pred[:, i]) for i in range(y_pred.shape[1])])
    except ValueError:
        roc_auc = 0.5
    log_loss_positive, log_loss_negative, log_loss_weighted = multiclass_weighted_log_loss(y_true, y_pred)

    scores = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'log_loss_positive': log_loss_positive,
        'log_loss_negative': log_loss_negative,
        'log_loss_weighted': log_loss_weighted
    }

    return scores
