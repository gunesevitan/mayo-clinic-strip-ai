import json
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import settings
import metrics


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'train_metadata.csv')
    df_train['binary_encoded_label'] = df_train['label'].map({'CE': 0, 'LAA': 1}).astype(np.uint8)

    model_paths = [
        'mil_densenet121_16_256',
        'mil_densenetblur121d_16_256',
        'mil_densenet169_16_256',
        'mil_efficientnetb2_16_256',
        'mil_efficientnetv2rwt_16_256',
        'mil_coatlitemini_16_224',
        'mil_poolformer24_16_224',
        'mil_swintinypatch4window7_16_224'
    ]

    logging.info(f'Model Scores')

    for model_path in model_paths:

        model_train_predictions = pd.read_csv(settings.MODELS / model_path / 'train_predictions.csv')
        df_train[f'{model_path}_predictions'] = model_train_predictions['predictions'].values
        oof_scores = metrics.binary_classification_scores(df_train['binary_encoded_label'], df_train[f'{model_path}_predictions'], threshold=0.5)

        predictions_mean = df_train[f'{model_path}_predictions'].mean()
        predictions_std = df_train[f'{model_path}_predictions'].std()
        predictions_min = df_train[f'{model_path}_predictions'].min()
        predictions_max = df_train[f'{model_path}_predictions'].max()
        logging.info(f'{model_path} - {json.dumps(oof_scores, indent=2)} - Predictions Mean: {predictions_mean:.4f} Std: {predictions_std:.4f} Min: {predictions_min:.4f} Max: {predictions_max:.4f}\n')

    logging.info(f'Blend Scores')
    blend_weights = {
        'mil_densenet121_16_256': 0.10,
        'mil_densenet169_16_256': 0.10,
        'mil_densenetblur121d_16_256': 0.25,
        'mil_efficientnetb2_16_256': 0.125,
        'mil_efficientnetv2rwt_16_256': 0.125,
        'mil_coatlitemini_16_224': 0.10,
        'mil_poolformer24_16_224': 0.10,
        'mil_swintinypatch4window7_16_224': 0.10
    }
    blend_predictions = np.zeros(df_train.shape[0])
    for model, weight in blend_weights.items():
        blend_predictions += (df_train[f'{model}_predictions'].values * weight)

    df_train['blend_predictions'] = blend_predictions
    blend_oof_scores = metrics.binary_classification_scores(df_train['binary_encoded_label'], df_train['blend_predictions'], threshold=0.5)
    predictions_mean = df_train[f'blend_predictions'].mean()
    predictions_std = df_train[f'blend_predictions'].std()
    predictions_min = df_train[f'blend_predictions'].min()
    predictions_max = df_train[f'blend_predictions'].max()
    logging.info(f'Blend - {json.dumps(blend_oof_scores, indent=2)} - Predictions Mean: {predictions_mean:.4f} Std: {predictions_std:.4f} Min: {predictions_min:.4f} Max: {predictions_max:.4f}\n')

    logging.info(f'Stack Scores')
    prediction_columns = [column for column in df_train.columns if column.endswith('predictions')]
    linear_model = LinearRegression()
    linear_model.fit(df_train[prediction_columns], df_train['binary_encoded_label'])
    stack_predictions = np.clip(linear_model.predict(df_train[prediction_columns]), a_min=0, a_max=1)
    df_train['stack_predictions'] = stack_predictions
    stack_oof_scores = metrics.binary_classification_scores(df_train['binary_encoded_label'], df_train['stack_predictions'], threshold=0.5)
    predictions_mean = df_train[f'stack_predictions'].mean()
    predictions_std = df_train[f'stack_predictions'].std()
    predictions_min = df_train[f'stack_predictions'].min()
    predictions_max = df_train[f'stack_predictions'].max()
    logging.info(f'Stack - {json.dumps(stack_oof_scores, indent=2)} - Predictions Mean: {predictions_mean:.4f} Std: {predictions_std:.4f} Min: {predictions_min:.4f} Max: {predictions_max:.4f}\n')

    logging.info('Adjusted Predictions Score')
    adjusted_prediction_column = 'blend_predictions'
    df_train[adjusted_prediction_column] += 0.21
    df_train_aggregated = df_train.groupby('patient_id')[[adjusted_prediction_column, 'binary_encoded_label']].mean()
    adjusted_oof_scores = metrics.binary_classification_scores(df_train_aggregated['binary_encoded_label'], df_train_aggregated[adjusted_prediction_column], threshold=0.5)
    predictions_mean = df_train[adjusted_prediction_column].mean()
    predictions_std = df_train[adjusted_prediction_column].std()
    predictions_min = df_train[adjusted_prediction_column].min()
    predictions_max = df_train[adjusted_prediction_column].max()
    logging.info(f'Adjusted Predictions - {json.dumps(adjusted_oof_scores, indent=2)} - Predictions Mean: {predictions_mean:.4f} Std: {predictions_std:.4f} Min: {predictions_min:.4f} Max: {predictions_max:.4f}\n')
