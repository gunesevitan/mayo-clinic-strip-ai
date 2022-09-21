import argparse
import yaml
import pandas as pd

import settings
import tabular_preprocessing
import torch_trainers


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)

    if config['task'] == 'binary_classification' or config['task'] == 'multiclass_classification':

        df_train = pd.read_csv(settings.DATA / 'train_metadata.csv')
        df_test = pd.read_csv(settings.DATA / 'test_metadata.csv')
        df_folds = pd.read_csv(settings.DATA / 'folds.csv')

        df_train, df_test = tabular_preprocessing.process_datasets(
            df_train=df_train,
            df_test=df_test,
            df_folds=df_folds
        )

        trainer = torch_trainers.ClassificationTrainer(
            dataset_parameters=config['dataset_parameters'],
            model_parameters=config['model_parameters'],
            training_parameters=config['training_parameters'],
            transform_parameters=config['transform_parameters'],
            inference_parameters=config['inference_parameters'],
            persistence_parameters=config['persistence_parameters']
        )

        if args.mode == 'train':
            trainer.train_and_validate(df_train=df_train)
        elif args.mode == 'inference':
            trainer.inference(df_train=df_train)
