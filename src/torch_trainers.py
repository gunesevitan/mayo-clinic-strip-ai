import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2, 40).__str__()
import logging
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

import settings
import image_utils
import visualization
import transforms
import torch_datasets
import torch_modules
import torch_utils
import metrics


class ClassificationTrainer:

    def __init__(self, dataset_parameters, model_parameters, training_parameters, transform_parameters, inference_parameters, persistence_parameters):

        self.dataset_parameters = dataset_parameters
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters
        self.transform_parameters = transform_parameters
        self.inference_parameters = inference_parameters
        self.persistence_parameters = persistence_parameters

    def train(self, train_loader, model, criterion, optimizer, device, scheduler=None):

        """
        Train given model on given data loader

        Parameters
        ----------
        train_loader (torch.utils.data.DataLoader): Training set data loader
        model (torch.nn.Module): Model to train
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Location of the model and inputs
        scheduler (torch.optim.LRScheduler or None): Learning rate scheduler

        Returns
        -------
        train_loss (float): Average training loss after model is fully trained on training set data loader
        """

        model.train()
        progress_bar = tqdm(train_loader)
        losses = []

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.detach().item())
            average_loss = np.mean(losses)
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            progress_bar.set_description(f'train_loss: {average_loss:.6f} - lr: {lr:.8f}')

        train_loss = np.mean(losses)
        return train_loss

    def validate(self, val_loader, model, criterion, device):

        """
        Validate given model on given data loader

        Parameters
        ----------
        val_loader (torch.utils.data.DataLoader): Validation set data loader
        model (torch.nn.Module): Model to validate
        criterion (torch.nn.Module): Loss function
        device (torch.device): Location of the model and inputs

        Returns
        -------
        val_loss (float): Average validation loss after model is fully validated on validation set data loader
        val_accuracy (float): Validation accuracy after model is fully validated on validation set data loader
        predictions (numpy.ndarray of shape (n_samples, 1)): Label predictions of the model
        """

        model.eval()
        progress_bar = tqdm(val_loader)
        losses = []
        ground_truth = []
        predictions = []

        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses.append(loss.detach().item())
                average_loss = np.mean(losses)
                progress_bar.set_description(f'val_loss: {average_loss:.6f}')
                ground_truth += [(targets.detach().cpu())]
                predictions += [(outputs.detach().cpu())]

        val_loss = np.mean(losses)
        ground_truth = torch.cat(ground_truth, dim=0).numpy()
        predictions = torch.cat(predictions, dim=0).numpy()

        if self.dataset_parameters['targets'] == 'binary_encoded_label':
            val_scores = metrics.binary_classification_scores(y_true=ground_truth, y_pred=predictions, threshold=0.5)
        elif self.dataset_parameters['targets'] == 'multiclass_encoded':
            val_scores = metrics.multiclass_classification_scores(y_true=ground_truth, y_pred=predictions)
        else:
            raise ValueError(f'Invalid targets {self.dataset_parameters["targets"]}')

        return val_loss, val_scores, predictions

    def train_and_validate(self, df_train):

        """
        Train and validate on inputs and targets listed on given dataframes with specified configuration and transforms

        Parameters
        ----------
        df_train (pandas.DataFrame of shape (n_rows, n_columns)): Dataframe of filenames, targets and folds
        """

        logging.info(f'\n{"-" * 30}\nRunning {self.persistence_parameters["name"]} Model for Training - Seed: {self.training_parameters["random_state"]}\n{"-" * 30}\n')

        # Create directory for models and visualizations
        model_root_directory = Path(settings.MODELS / self.persistence_parameters['name'])
        model_root_directory.mkdir(parents=True, exist_ok=True)

        dataset_transforms = transforms.get_classification_transforms(**self.transform_parameters)
        scores = []

        for fold in self.training_parameters['folds']:

            train_idx, val_idx = df_train.loc[df_train[fold] == 0].index, df_train.loc[df_train[fold] == 1].index
            # Validate on training set if validation is set is not specified
            if len(val_idx) == 0:
                val_idx = train_idx

            logging.info(f'\nTraining: {len(train_idx)} ({len(train_idx) // self.training_parameters["training_batch_size"] + 1} steps) - Validation {len(val_idx)} ({len(val_idx) // self.training_parameters["test_batch_size"] + 1} steps)')
            train_dataset = torch_datasets.ClassificationDataset(
                image_ids=df_train.loc[train_idx, 'image_id'].values,
                labels=df_train.loc[train_idx, self.dataset_parameters['targets']].values,
                n_tiles=self.dataset_parameters['n_tiles'],
                transforms=dataset_transforms['train'],
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_parameters['training_batch_size'],
                sampler=RandomSampler(train_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=self.training_parameters['num_workers']
            )
            val_dataset = torch_datasets.ClassificationDataset(
                image_ids=df_train.loc[val_idx, 'image_id'].values,
                labels=df_train.loc[val_idx, self.dataset_parameters['targets']].values,
                n_tiles=self.dataset_parameters['n_tiles'],
                transforms=dataset_transforms['test'],
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_parameters['test_batch_size'],
                sampler=SequentialSampler(val_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=self.training_parameters['num_workers']
            )

            # Set model, loss function, device and seed for reproducible results
            torch_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
            device = torch.device(self.training_parameters['device'])
            criterion = getattr(torch_modules, self.training_parameters['loss_function'])(**self.training_parameters['loss_args'])

            model = getattr(torch_modules, self.model_parameters['model_class'])(**self.model_parameters['model_args'])
            if self.model_parameters['model_checkpoint_path'] is not None:
                model.load_state_dict(torch.load(self.model_parameters['model_checkpoint_path']))
            model.to(device)

            # Set optimizer, learning rate scheduler and stochastic weight averaging
            optimizer = getattr(optim, self.training_parameters['optimizer'])(model.parameters(), **self.training_parameters['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, self.training_parameters['lr_scheduler'])(optimizer, **self.training_parameters['lr_scheduler_args'])

            early_stopping = False
            summary = {
                'train_loss': [],
                'val_loss': [],
                'val_accuracy': [],
                'val_roc_auc': [],
                'val_log_loss': []
            }

            for epoch in range(1, self.training_parameters['epochs'] + 1):

                if early_stopping:
                    break

                if self.training_parameters['lr_scheduler'] == 'ReduceLROnPlateau':
                    # Step on validation loss if learning rate scheduler is ReduceLROnPlateau
                    train_loss = self.train(train_loader, model, criterion, optimizer, device, scheduler=None)
                    val_loss, val_scores, val_predictions = self.validate(val_loader, model, criterion, device)
                    scheduler.step(val_loss)
                else:
                    # Learning rate scheduler works in training function if it is not ReduceLROnPlateau
                    train_loss = self.train(train_loader, model, criterion, optimizer, device, scheduler)
                    val_loss, val_scores, val_predictions = self.validate(val_loader, model, criterion, device)

                logging.info(
                    f'''
                    Epoch {epoch} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}
                    Validation Accuracy: {val_scores["accuracy"]:.4f} ROC AUC: {val_scores["roc_auc"]:.4f} Log Loss: {val_scores["log_loss"]:.4f}
                    '''
                )

                if epoch in self.persistence_parameters['save_epoch_model']:
                    # Save model if current epoch is specified to be saved
                    torch.save(model.state_dict(), model_root_directory / f'model_{fold}_epoch_{epoch}.pt')
                    logging.info(f'Saved model_{fold}_epoch_{epoch}.pt to {model_root_directory}')

                best_val_loss = np.min(summary['val_loss']) if len(summary['val_loss']) > 0 else np.inf
                if val_loss < best_val_loss:
                    # Save model if validation loss improves
                    torch.save(model.state_dict(), model_root_directory / f'model_{fold}_best.pt')
                    print(f'Saved model_{fold}_best.pt (validation loss decreased from {best_val_loss:.6f} to {val_loss:.6f})\n')
                    df_train.loc[val_idx, 'predictions'] = val_predictions

                summary['train_loss'].append(train_loss)
                summary['val_loss'].append(val_loss)
                summary['val_accuracy'].append(val_scores['accuracy'])
                summary['val_roc_auc'].append(val_scores['roc_auc'])
                summary['val_log_loss'].append(val_scores['log_loss'])

                best_epoch = np.argmin(summary['val_loss'])
                if self.training_parameters['early_stopping_patience'] > 0:
                    # Trigger early stopping if early stopping patience is greater than 0
                    if len(summary['val_loss']) - best_epoch >= self.training_parameters['early_stopping_patience']:
                        logging.info(
                            f'''
                            Early Stopping (validation loss didn\'t improve for {self.training_parameters["early_stopping_patience"]} epochs)
                            Best Epoch ({best_epoch + 1}) Validation Loss: {summary["val_loss"][best_epoch]:.4f} Accuracy: {summary["val_accuracy"][best_epoch]:.4f} ROC AUC: {summary["val_roc_auc"][best_epoch]:.4f} Log Loss: {summary["val_log_loss"][best_epoch]:.4f}'
                            '''
                        )
                        early_stopping = True
                        scores.append({
                            'val_loss': summary['val_loss'][best_epoch],
                            'val_accuracy': summary['val_accuracy'][best_epoch],
                            'val_roc_auc': summary['val_roc_auc'][best_epoch],
                            'val_log_loss': summary['val_log_loss'][best_epoch],
                        })
                else:
                    if epoch == self.training_parameters['epochs']:
                        scores.append({
                            'val_loss': summary['val_loss'][best_epoch],
                            'val_accuracy': summary['val_accuracy'][best_epoch],
                            'val_roc_auc': summary['val_roc_auc'][best_epoch],
                            'val_log_loss': summary['val_log_loss'][best_epoch],
                        })

            if self.persistence_parameters['visualize_learning_curve']:
                visualization.visualize_learning_curve(
                    training_losses=summary['train_loss'],
                    validation_losses=summary['val_loss'],
                    validation_scores={
                        'val_accuracy': summary['val_accuracy'],
                        'val_roc_auc': summary['val_roc_auc'],
                        'val_log_loss': summary['val_log_loss']
                    },
                    path=model_root_directory / f'learning_curve_{fold}.png'
                )
                logging.info(f'Saved learning_curve_{fold}.png to {model_root_directory}')

        df_scores = pd.DataFrame(scores)
        for score_idx, row in df_scores.iterrows():
            logging.info(f'Validation Scores: {json.dumps(row.to_dict(), indent=2)}')
        logging.info(f'\nMean Validation Scores: {json.dumps(df_scores.mean(axis=0).to_dict(), indent=2)} (±{json.dumps(df_scores.std(axis=0).to_dict(), indent=2)})')

        if self.dataset_parameters['targets'] == 'binary_encoded_label':
            oof_scores = metrics.binary_classification_scores(
                y_true=df_train[self.dataset_parameters['targets']],
                y_pred=df_train['predictions'],
                threshold=0.5
            )
        elif self.dataset_parameters['targets'] == 'multiclass_encoded':
            oof_scores = metrics.multiclass_classification_scores(
                y_true=df_train[self.dataset_parameters['targets']],
                y_pred=df_train['predictions']
            )
        else:
            raise ValueError(f'Invalid targets {self.dataset_parameters["targets"]}')

        logging.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

        if self.persistence_parameters['visualize_training_scores']:
            visualization.visualize_scores(
                df_scores=df_scores,
                path=model_root_directory / f'training_scores.png'
            )
            logging.info(f'Saved training_scores.png to {model_root_directory}')

    def inference(self, df_train):

        """
        Inference on inputs and targets listed on given dataframes with specified configuration and transforms

        Parameters
        ----------
        df_train (pandas.DataFrame of shape (n_rows, n_columns)): Dataframe of filenames, targets and folds
        """

        logging.info(f'\n{"-" * 30}\nRunning {self.persistence_parameters["name"]} Model for Inference - Seed: {self.training_parameters["random_state"]}\n{"-" * 30}\n')

        # Create directory for models and visualizations
        model_root_directory = Path(settings.MODELS / self.persistence_parameters['name'])
        model_root_directory.mkdir(parents=True, exist_ok=True)

        test_transforms = transforms.get_classification_transforms(**self.transform_parameters)['test']
        scores = {
            'fold_scores': {},
            'oof_scores': None
        }

        for fold in self.inference_parameters['folds']:

            val_idx = df_train.loc[df_train[fold] == 1].index
            logging.info(f'\n{fold}  - Validation {len(val_idx)} ({len(val_idx) // self.training_parameters["test_batch_size"] + 1} steps)')

            # Set model, loss function, device and seed for reproducible results
            torch_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
            device = torch.device(self.training_parameters['device'])

            model = getattr(torch_modules, self.model_parameters['model_class'])(**self.model_parameters['model_args'])
            model.load_state_dict(torch.load(model_root_directory / f'model_{fold}_best.pt'))
            model.to(device)
            model.eval()

            for idx, row in tqdm(df_train.loc[val_idx, :].iterrows(), total=len(val_idx)):

                tiles = []
                for tile_idx in range(self.dataset_parameters['n_tiles']):
                    image = cv2.imread(str(settings.DATA / 'train_compressed_tiles' / f'{row["image_id"]}_{tile_idx}.jpg'))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    tiles.append(image)

                # Apply transforms to tiles and stack them along the batch dimension
                tiles = [test_transforms(image=tile)['image'].float() for tile in tiles]
                tiles = torch.stack(tiles, dim=0)
                inputs = torch.unsqueeze(tiles, dim=0)
                inputs = inputs.to('cuda')

                with torch.no_grad():
                    outputs = model(inputs)

                predictions = outputs.detach().cpu()
                predictions = torch.sigmoid(torch.squeeze(predictions, dim=1)).numpy().astype(np.float32)
                df_train.loc[idx, 'predictions'] = predictions

            if self.dataset_parameters['targets'] == 'binary_encoded_label':
                fold_scores = metrics.binary_classification_scores(
                    y_true=df_train.loc[val_idx, 'binary_encoded_label'],
                    y_pred=df_train.loc[val_idx, 'predictions'],
                    threshold=0.5
                )
            elif self.dataset_parameters['targets'] == 'multiclass_encoded':
                fold_scores = metrics.multiclass_classification_scores(
                    y_true=df_train.loc[val_idx, 'multiclass_encoded'],
                    y_pred=df_train.loc[val_idx, 'predictions']
                )
            else:
                raise ValueError(f'Invalid targets {self.dataset_parameters["targets"]}')

            logging.info(f'Validation Scores: {json.dumps(fold_scores, indent=2)}')
            scores['fold_scores'][fold] = fold_scores

        if self.dataset_parameters['targets'] == 'binary_encoded_label':
            oof_scores = metrics.binary_classification_scores(
                y_true=df_train.loc[:, 'binary_encoded_label'],
                y_pred=df_train.loc[:, 'predictions'],
                threshold=0.5
            )
        elif self.dataset_parameters['targets'] == 'multiclass_encoded':
            oof_scores = metrics.multiclass_classification_scores(
                y_true=df_train.loc[:, 'multiclass_encoded'],
                y_pred=df_train.loc[:, 'predictions']
            )
        else:
            raise ValueError(f'Invalid targets {self.dataset_parameters["targets"]}')

        logging.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')
        scores['oof_scores'] = oof_scores
        with open(model_root_directory / 'inference_scores.json', mode='w') as f:
            json.dump(scores, f, indent=2)
        logging.info(f'Saved inference_scores.json to {model_root_directory}')

        df_train[['image_id', 'predictions']].to_csv(model_root_directory / 'train_predictions.csv', index=False)
        logging.info(f'Saved train_predictions.csv to {model_root_directory}')
