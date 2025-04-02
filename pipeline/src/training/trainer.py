import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import uuid
from tqdm import tqdm

from ..utils.config import TrainingConfig
from ..utils.metrics import calculate_metrics
from ..utils.model_factory import create_binding_predictor, create_glycan_encoder, create_protein_encoder
from ..data.dataset import prepare_train_val_datasets

class ExperimentTracker:
    """
    Class to track experiment results across multiple runs with different configurations.
    """
    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)
        self.results_file = self.base_dir / "experiment_results.csv"
        
        # Create the results file if it doesn't exist
        if not self.results_file.exists():
            self._create_results_file()
        
    def _create_results_file(self):
        """Initialize the results file with column headers."""
        # Make sure the directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create empty dataframe with appropriate columns
        columns = [
            'experiment_id',
            'glycan_encoder_type',
            'protein_encoder_type',
            'binding_predictor_type',
            'batch_size',
            'learning_rate',
            'log_predict',
            'split_mode',
            'use_kfold',
            'k_folds',
            'best_train_loss',
            'best_val_loss',
            'best_train_mse',
            'best_val_mse',
            'best_train_pearson',
            'best_val_pearson',
            'best_epoch',
            'timestamp'
        ]
        
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.results_file, index=False)
        
    def add_experiment_result(self, experiment_id, config, metrics_df):
        """
        Add results from a completed experiment to the results table.
        
        Args:
            experiment_id: Unique identifier for the experiment
            config: TrainingConfig object with experiment settings
            metrics_df: DataFrame containing the metrics history
        """
        # Find the best validation metrics (usually the lowest validation loss)
        best_idx = metrics_df['val_loss'].idxmin()
        best_metrics = metrics_df.iloc[best_idx]
        
        # Create a new row for the results table
        result = {
            'experiment_id': experiment_id,
            'glycan_encoder_type': config.glycan_encoder_type,
            'protein_encoder_type': config.protein_encoder_type,
            'binding_predictor_type': config.binding_predictor_type,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'log_predict': config.log_predict,
            'split_mode': config.split_mode,
            'use_kfold': config.use_kfold,
            'k_folds': config.k_folds if config.use_kfold else None,
            'best_train_loss': best_metrics['train_loss'],
            'best_val_loss': best_metrics['val_loss'],
            'best_train_mse': best_metrics['train_mse'],
            'best_val_mse': best_metrics['val_mse'],
            'best_train_pearson': best_metrics['train_pearson'],
            'best_val_pearson': best_metrics['val_pearson'],
            'best_epoch': best_metrics['epoch'],
            'timestamp': best_metrics['timestamp']
        }
        
        # Load existing results
        try:
            results_df = pd.read_csv(self.results_file)
        except Exception:
            # If there's an issue with the file, recreate it
            self._create_results_file()
            results_df = pd.read_csv(self.results_file)
        
        # Add the new result and save
        results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
        results_df.to_csv(self.results_file, index=False)
        
        return results_df
    
    def get_results_table(self):
        """Get the current results table as a DataFrame."""
        try:
            return pd.read_csv(self.results_file)
        except Exception:
            self._create_results_file()
            return pd.read_csv(self.results_file)

class GlycoProteinDataset(Dataset):
    def __init__(self, fractions_df, glycan_encodings, protein_encodings, glycan_mapping, protein_mapping):
        """
        Args:
            fractions_df: DataFrame with fraction data
            glycan_encodings: Tensor of shape [n_glycans, embedding_dim]
            protein_encodings: Tensor of shape [n_proteins, embedding_dim]
            glycan_mapping: Dict mapping glycan IDs to indices in glycan_encodings
            protein_mapping: Dict mapping protein IDs to indices in protein_encodings
        """
        self.fractions_df = fractions_df
        self.glycan_encodings = glycan_encodings
        self.protein_encodings = protein_encodings
        self.glycan_mapping = glycan_mapping
        self.protein_mapping = protein_mapping
        
    def __len__(self):
        return len(self.fractions_df)
    
    def __getitem__(self, idx):
        row = self.fractions_df.iloc[idx]
        
        # Get the corresponding encodings using the mappings
        glycan_idx = self.glycan_mapping[row['GlycanID']]
        protein_idx = self.protein_mapping[row['ProteinGroup']]
        
        return {
            'glycan_encoding': self.glycan_encodings[glycan_idx],
            'protein_encoding': self.protein_encodings[protein_idx],
            'concentration': torch.tensor([row['Concentration']], dtype=torch.float32),
            'target': torch.tensor([row['f']], dtype=torch.float32)
        }
        
# https://stackoverflow.com/a/74801406
class weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights):
        return ((inputs - targets)**2 ) * weights

class BindingTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        torch.manual_seed(self.config.random_state)
        
        self.experiment_dir = Path(config.output_dir)

        self.experiment_id = str(uuid.uuid4())[:8]  # Using first 8 chars of UUID
        base_dir = str(Path(config.output_dir).parent)  # Use parent of the experiment dir for results table
        self.tracker = ExperimentTracker(base_dir)

        self.setup_models()
    
    def setup_models(self):
        # create the models using the factory functions
        self.glycan_encoder = create_glycan_encoder(
            self.config.glycan_encoder_type,
        ).to(self.device)
        
        self.protein_encoder = create_protein_encoder(
            self.config.protein_encoder_type,
        ).to(self.device)
        
        # build the param dictionary for the binding predictor
        predictor_params = {
            'glycan_dim': self.glycan_encoder.embedding_dim,
            'protein_dim': self.protein_encoder.embedding_dim
        }
        if 'dnn_hidden_dims' in self.config.model_specific_params:
            predictor_params['hidden_dims'] = self.config.model_specific_params['dnn_hidden_dims']
        
        self.binding_predictor = create_binding_predictor(
            self.config.binding_predictor_type,
            **predictor_params
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            list(self.glycan_encoder.parameters()) +
            list(self.protein_encoder.parameters()) +
            list(self.binding_predictor.parameters()),
            lr=self.config.learning_rate
        )
        self.criterion = weighted_MSELoss() #self.weighted_mse_loss #nn.MSELoss() 
        
    def weighted_mse_loss(self, predictions, targets, weight=None):
        """
        Weighted MSE loss
        
        Args:
            predictions: Predicted values
            targets: Target values
            weight: Optional weight factor
        """
        if weight is None:
            return nn.MSELoss()(predictions, targets)
        else:
            return (weight * ((predictions - targets) ** 2)).mean()
        
    def reset_models(self):
        """Reset all models to their initial state"""
        self.setup_models()
    
    def _train_epoch(self, train_loader: DataLoader, fold_weight: int) -> Dict[str, float]:
        self.glycan_encoder.train()
        self.protein_encoder.train()
        self.binding_predictor.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            glycan_encoding = batch['glycan_encoding'].to(self.device)
            protein_encoding = batch['protein_encoding'].to(self.device)
            concentration = batch['concentration'].to(self.device)
            
            targets = batch['target'].to(self.device)
            
            if self.config.log_predict:
                targets = torch.log(targets + 1e-6)
            
            predictions = self.binding_predictor(
                glycan_encoding,
                protein_encoding,
                concentration
            )
            
            loss = self.criterion(predictions, targets, fold_weight)
            
            # average out loss across the batch
            loss = loss.mean()
            
            # reset gradients to zero
            self.optimizer.zero_grad()
            # perform backpropigation to calculate the gradients we need to improve model
            loss.backward(retain_graph=True)
            # update the weights with our loss gradients
            self.optimizer.step()
            
            # revert predictions and targets to original values for original analysis if using log transform
            if self.config.log_predict:
                predictions = torch.exp(predictions) - 1e-6
                targets = torch.exp(targets) - 1e-6
            
            # track totals
            total_loss += loss.item()
            all_predictions.append(predictions.detach())
            all_targets.append(targets.detach())
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # save metrics
        epoch_predictions = torch.cat(all_predictions)
        epoch_targets = torch.cat(all_targets)
        metrics = calculate_metrics(epoch_predictions, epoch_targets)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics, epoch_predictions, epoch_targets
    
    def _validate(self, val_loader: DataLoader, fold_weight: int) -> Dict[str, float]:
        self.glycan_encoder.eval()
        self.protein_encoder.eval()
        self.binding_predictor.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for batch in pbar:
                glycan_encoding = batch['glycan_encoding'].to(self.device)
                protein_encoding = batch['protein_encoding'].to(self.device)
                concentration = batch['concentration'].to(self.device)
                targets = batch['target'].to(self.device)
                
                if self.config.log_predict:
                    targets = torch.log(targets + 1e-6)
                
                predictions = self.binding_predictor(
                    glycan_encoding,
                    protein_encoding,
                    concentration
                )

                loss = self.criterion(predictions, targets, fold_weight).mean()
                
                # revert predictions and targets to original values for original analysis if using log transform
                if self.config.log_predict:
                    predictions = torch.exp(predictions) - 1e-6
                    targets = torch.exp(targets) - 1e-6
                
                # track totals
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(targets)
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # save metrics
        val_predictions = torch.cat(all_predictions)
        val_targets = torch.cat(all_targets)
        metrics = calculate_metrics(val_predictions, val_targets)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics, val_predictions, val_targets
    
    def save_checkpoint(self, fold: int = None, epoch: int = None):
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html

        checkpoint = {
            'glycan_encoder': self.glycan_encoder.state_dict(),
            'protein_encoder': self.protein_encoder.state_dict(),
            'binding_predictor': self.binding_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        
        checkpoint_dir = self.experiment_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        if fold is not None and epoch is not None:
            checkpoint_name = f"model_fold_{fold}_epoch_{epoch}.pt"
        elif fold is not None:
            checkpoint_name = f"model_fold_{fold}_final.pt"
        elif epoch is not None:
            checkpoint_name = f"model_epoch_{epoch}.pt"
        else:
            checkpoint_name = "model_final.pt"
            
        torch.save(
            checkpoint,
            checkpoint_dir / checkpoint_name
        )
    
    def create_error_histogram(self, predictions, targets, prefix="", train_predictions=None, train_targets=None):
        """
        Create histograms of prediction errors: 
        1. Stacked raw error distribution with best fit line
        2. Stacked absolute error histogram (without best fit line)
        
        Args:
            predictions: Tensor of predictions (validation set)
            targets: Tensor of target values (validation set)
            prefix: Optional prefix for the filename (e.g., 'train', 'val', 'final')
            train_predictions: Optional tensor of training predictions for comparison
            train_targets: Optional tensor of training targets for comparison
        """
        prefix = f"{self.config.protein_encoder_type}_{self.config.glycan_encoder_type}_{self.config.binding_predictor_type}"

        # Convert tensors to numpy
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        
        # Calculate errors (absolute and raw)
        absolute_errors = np.abs(predictions_np - targets_np)
        raw_errors = predictions_np - targets_np  # For distribution with negative values
        
        # Process training data if provided
        has_training_data = train_predictions is not None and train_targets is not None
        
        if has_training_data:
            train_predictions_np = train_predictions.cpu().numpy().flatten()
            train_targets_np = train_targets.cpu().numpy().flatten()
            train_absolute_errors = np.abs(train_predictions_np - train_targets_np)
            train_raw_errors = train_predictions_np - train_targets_np
        
        # Determine bin parameters
        bins = 30
        
        # Save dir
        plots_dir = self.experiment_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Absolute Error Histogram (Stacked, no best fit line)
        plt.figure(figsize=(10, 6))
        
        # Calculate the maximum range for the bins
        max_error = absolute_errors.max()
        if has_training_data:
            max_error = max(max_error, train_absolute_errors.max())
        
        # Create bins with a little buffer
        hist_bins = np.linspace(0, max_error * 1.05, bins)
        
        if has_training_data:
            # Create a stacked histogram
            plt.hist([absolute_errors, train_absolute_errors], bins=hist_bins, 
                    label=['Validation', 'Training'], alpha=0.7, 
                    color=['skyblue', 'teal'], edgecolor=['navy', 'darkgreen'],
                    stacked=True)
        else:
            # Plot just validation histogram
            plt.hist(absolute_errors, bins=hist_bins, alpha=0.7, 
                    color='skyblue', edgecolor='navy', label='Validation')
        
        plt.xlabel('Absolute Error')
        plt.ylabel('Count')
        plt.title(f'{prefix} Absolute Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add stats to the plot
        mean_error = np.mean(absolute_errors)
        median_error = np.median(absolute_errors)
        
        if has_training_data:
            train_mean_error = np.mean(train_absolute_errors)
            train_median_error = np.median(train_absolute_errors)
            combined_abs_errors = np.concatenate([absolute_errors, train_absolute_errors])
            combined_mean = np.mean(combined_abs_errors)
            combined_median = np.median(combined_abs_errors)
            stats_text = f'Combined Mean: {combined_mean:.4f}\nCombined Median: {combined_median:.4f}'
        else:
            stats_text = f'Mean: {mean_error:.4f}\nMedian: {median_error:.4f}'
        
        plt.annotate(stats_text, xy=(0.65, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Save the absolute error plot
        plt.savefig(plots_dir / f'{prefix}_absolute_error.png')
        plt.close()
        
        # 2. Raw Error Distribution (Stacked with negative values and best fit line)
        plt.figure(figsize=(10, 6))
        
        # Calculate the maximum range for the bins
        min_raw_error = raw_errors.min()
        max_raw_error = raw_errors.max()
        
        if has_training_data:
            min_raw_error = min(min_raw_error, train_raw_errors.min())
            max_raw_error = max(max_raw_error, train_raw_errors.max())
        
        # Create bins
        raw_hist_bins = np.linspace(min_raw_error * 1.05, max_raw_error * 1.05, bins)
        
        if has_training_data:
            # Create a stacked histogram for raw errors
            plt.hist([raw_errors, train_raw_errors], bins=raw_hist_bins, 
                    label=['Validation', 'Training'], alpha=0.7, 
                    color=['skyblue', 'teal'], edgecolor=['navy', 'darkgreen'],
                    stacked=True)
            
            # Add a single KDE curve for combined data
            from scipy.stats import gaussian_kde
            x_range = np.linspace(min_raw_error * 1.05, max_raw_error * 1.05, 1000)
            combined_raw_errors = np.concatenate([raw_errors, train_raw_errors])
            
            if len(combined_raw_errors) > 1:
                combined_kde = gaussian_kde(combined_raw_errors)
                plt.plot(x_range, combined_kde(x_range) * len(combined_raw_errors) * (raw_hist_bins[1] - raw_hist_bins[0]), 
                        'r-', linewidth=2, label='Error Distribution')
        else:
            # Plot just validation histogram
            plt.hist(raw_errors, bins=raw_hist_bins, alpha=0.7, 
                    color='skyblue', edgecolor='navy', label='Validation')
            
            if len(raw_errors) > 1:
                from scipy.stats import gaussian_kde
                x_range = np.linspace(min_raw_error * 1.05, max_raw_error * 1.05, 1000)
                val_kde = gaussian_kde(raw_errors)
                plt.plot(x_range, val_kde(x_range) * len(raw_errors) * (raw_hist_bins[1] - raw_hist_bins[0]), 
                        'r-', linewidth=2, label='Error Distribution')
        
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        plt.xlabel('Error (Predicted - Actual)')
        plt.ylabel('Count')
        plt.title(f'{prefix} Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add stats
        if has_training_data:
            mean_raw_error = np.mean(combined_raw_errors)
            std_raw_error = np.std(combined_raw_errors)
        else:
            mean_raw_error = np.mean(raw_errors)
            std_raw_error = np.std(raw_errors)
        
        stats_text = f'Mean: {mean_raw_error:.4f}\nStd Dev: {std_raw_error:.4f}'
        plt.annotate(stats_text, xy=(0.05, 0.75), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Save the raw error plot
        plt.savefig(plots_dir / f'{prefix}_error_distribution.png')
        plt.close()
        
        # Return statistics for logging
        result = {
            'mean_abs_error': mean_error if not has_training_data else combined_mean,
            'median_abs_error': median_error if not has_training_data else combined_median,
            'mean_raw_error': mean_raw_error,
            'std_raw_error': std_raw_error
        }
            
        return result

    def plot_metrics(self, metrics_df: pd.DataFrame, fold_metrics_df: pd.DataFrame = None):
        plots_dir = self.experiment_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Plot average metrics across folds
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
        plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Average Training and Validation Loss Across Folds')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'avg_loss_curves.png')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epoch'], metrics_df['train_mse'], label='Train MSE')
        plt.plot(metrics_df['epoch'], metrics_df['val_mse'], label='Validation MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Average Training and Validation MSE Across Folds')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'avg_mse_curves.png')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epoch'], metrics_df['train_pearson'], label='Train Pearson')
        plt.plot(metrics_df['epoch'], metrics_df['val_pearson'], label='Validation Pearson')
        plt.xlabel('Epoch')
        plt.ylabel('Pearson Correlation')
        plt.title('Average Training and Validation Pearson Correlation Across Folds')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'avg_pearson_curves.png')
        plt.close()
        
        # If fold-specific metrics are provided, plot them too
        if fold_metrics_df is not None:
            # Plot loss per fold
            plt.figure(figsize=(10, 6))
            for fold in fold_metrics_df['fold'].unique():
                fold_data = fold_metrics_df[fold_metrics_df['fold'] == fold]
                plt.plot(fold_data['epoch'], fold_data['val_loss'], label=f'Fold {fold}')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.title('Validation Loss by Fold')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / 'fold_loss_curves.png')
            plt.close()
            
            # Plot MSE per fold
            plt.figure(figsize=(10, 6))
            for fold in fold_metrics_df['fold'].unique():
                fold_data = fold_metrics_df[fold_metrics_df['fold'] == fold]
                plt.plot(fold_data['epoch'], fold_data['val_mse'], label=f'Fold {fold}')
            plt.xlabel('Epoch')
            plt.ylabel('Validation MSE')
            plt.title('Validation MSE by Fold')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / 'fold_mse_curves.png')
            plt.close()
            
            # Plot Pearson correlation per fold
            plt.figure(figsize=(10, 6))
            for fold in fold_metrics_df['fold'].unique():
                fold_data = fold_metrics_df[fold_metrics_df['fold'] == fold]
                plt.plot(fold_data['epoch'], fold_data['val_pearson'], label=f'Fold {fold}')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Pearson Correlation')
            plt.title('Validation Pearson Correlation by Fold')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / 'fold_pearson_curves.png')
            plt.close()
    
    def save_worst_predictions(self, samples_df: pd.DataFrame, predictions: torch.Tensor, targets: torch.Tensor, num_samples: int = 10):
        """
        Save the worst sample predictions (based on highest absolute error) to an Excel file.
        
        Args:
            samples_df: DataFrame corresponding to the validation samples (must include identifying columns, e.g. GlycanID, ProteinGroup, etc.)
            predictions: Tensor of predictions (from validation)
            targets: Tensor of target values (from validation)
            num_samples: Number of worst samples to save (default is 10)
        """
        # Convert tensors to numpy arrays
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        
        # Calculate absolute errors
        absolute_errors = np.abs(predictions_np - targets_np)
        
        # Combine the validation DataFrame with predictions, targets, and computed errors
        worst_df = samples_df.copy()
        worst_df['Prediction'] = predictions_np
        worst_df['Target'] = targets_np
        worst_df['Absolute_Error'] = absolute_errors
        
        # Sort by absolute error in descending order and select the worst samples
        worst_df = worst_df.sort_values(by='Absolute_Error', ascending=False).head(num_samples)
        
        # Save the results to an Excel file with a sheet called "Worst Predictions"
        
        output_file = self.experiment_dir / 'worst_predictions.csv'
        worst_df.to_csv(output_file, index=False)

    def train(self, fractions_df: pd.DataFrame, glycans_df: pd.DataFrame, proteins_df: pd.DataFrame):
        # common metrics tracking
        all_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'train_pearson': [],
            'val_pearson': [],
            'epoch': [],
            'timestamp': []
        }
        
        fold_indices, glycan_encodings, protein_encodings = prepare_train_val_datasets(
            fractions_df,
            glycans_df,
            proteins_df,
            self.glycan_encoder,
            self.protein_encoder,
            self.config.random_state,
            self.config.split_mode,
            self.config.use_kfold,
            self.config.k_folds,
            self.config.val_split,
            self.config.device
        )
        
        # Create mappings
        glycan_mapping = {name: idx for idx, name in enumerate(glycans_df['Name'])}
        protein_mapping = {name: idx for idx, name in enumerate(proteins_df['ProteinGroup'])}
        
        # Tracking metrics
        fold_metrics = []
        fold_specific_metrics = []
        
        if self.config.use_kfold:
            print(f"Starting {self.config.k_folds}-fold cross-validation on our total {len(fractions_df)} samples")
            
            # Calculate fold weights
            train_sizes = [len(train_samples) for train_samples, test_samples in fold_indices]
            total_train_samples = sum(train_sizes)
            fold_weights = [total_train_samples / (len(fold_indices) * train_size) for train_size in train_sizes]
        else:
            print(f"Starting regular training with {len(fold_indices[0][0])} training samples and {len(fold_indices[0][1])} validation samples")
            fold_weights = [1.0]  # No special weighting for regular training
        
        # For each fold (or single split for regular training)
        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
            
            if len(test_idx) == 0:
                print('Test set size empty so skipping (try different split or k-fold)')
                continue
            
            # Get data for this fold/split
            train_data = fractions_df.loc[train_idx]
            val_data = fractions_df.loc[test_idx]
            fold_weight = fold_weights[fold_idx]
            
            train_pytorch_dataset = GlycoProteinDataset(
                train_data, glycan_encodings, protein_encodings, glycan_mapping, protein_mapping
            )
            val_pytorch_dataset = GlycoProteinDataset(
                val_data, glycan_encodings, protein_encodings, glycan_mapping, protein_mapping
            )
            
            if self.config.use_kfold:
                print(f"\n{'='*20} Fold {fold_idx+1}/{self.config.k_folds} {'='*20}")
            else:
                print(f"\n{'='*20} Training Run {'='*20}")
            
            # Reset models for this fold/run
            self.reset_models()
            
            # Create data loaders
            train_loader = DataLoader(
                train_pytorch_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_pytorch_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
            )
            
            print(f"Training with {len(train_data)} samples: ({(len(train_data) / (len(train_data) + len(val_data))) * 100:.2f}%), "
                  f"validating with {len(val_data)} samples: ({(len(val_data) / (len(train_data) + len(val_data))) * 100:.2f}%)")
                  
            if self.config.use_kfold:
                print(f"Fold weight: ~{fold_weight:.5f}")
            
            # Training loop for this fold/run
            fold_train_metrics = []
            fold_val_metrics = []
            
            for epoch in range(self.config.num_epochs):
                print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
                
                # if no k-fold then fold_weight is just 1
                train_metrics, train_predictions, train_targets = self._train_epoch(train_loader, fold_weight)
                val_metrics, val_predictions, val_targets = self._validate(val_loader, fold_weight)
                
                # Create histograms for the last epoch and save worst predictions
                if epoch == self.config.num_epochs - 1:
                    error_stats = self.create_error_histogram(
                        val_predictions, val_targets, 
                        prefix=f"fold{fold_idx}_final",
                        train_predictions=train_predictions, 
                        train_targets=train_targets
                    )
                    # Save the worst predictions from the validation set (uses the original val_data DataFrame)
                    self.save_worst_predictions(val_data, val_predictions, val_targets)
                
                # Save metrics for this fold/run
                timestamp = datetime.now().isoformat()
                fold_train_metrics.append(train_metrics)
                fold_val_metrics.append(val_metrics)
                
                # Save fold-specific metrics for visualization
                fold_specific_metrics.append({
                    'fold': fold_idx,
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_mse': train_metrics['mse'],
                    'val_mse': val_metrics['mse'],
                    'train_pearson': train_metrics['pearson'],
                    'val_pearson': val_metrics['pearson'],
                    'timestamp': timestamp
                })
                
                print(f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}")
                
                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    if self.config.use_kfold:
                        self.save_checkpoint(fold=fold_idx, epoch=epoch+1)
                    else:
                        self.save_checkpoint(epoch=epoch+1)
            
            # Save final model for this fold/run
            if self.config.use_kfold:
                self.save_checkpoint(fold=fold_idx)
            else:
                self.save_checkpoint()
            
            # Record metrics from this fold/run
            fold_metrics.append({
                'fold': fold_idx,
                'train_metrics': fold_train_metrics,
                'val_metrics': fold_val_metrics
            })
            # For k-fold, calculate average metrics across all folds
        # For regular training, this will just process the single "fold"
        for epoch in range(self.config.num_epochs):
            epoch_metrics = {
                'train_loss': 0,
                'val_loss': 0,
                'train_mse': 0,
                'val_mse': 0,
                'train_pearson': 0,
                'val_pearson': 0
            }
            
            # Sum metrics from all folds for this epoch
            for fold_data in fold_metrics:
                epoch_metrics['train_loss'] += fold_data['train_metrics'][epoch]['loss']
                epoch_metrics['val_loss'] += fold_data['val_metrics'][epoch]['loss']
                epoch_metrics['train_mse'] += fold_data['train_metrics'][epoch]['mse']
                epoch_metrics['val_mse'] += fold_data['val_metrics'][epoch]['mse']
                epoch_metrics['train_pearson'] += fold_data['train_metrics'][epoch]['pearson']
                epoch_metrics['val_pearson'] += fold_data['val_metrics'][epoch]['pearson']
            
            # Calculate average
            folds_count = len(fold_metrics)
            for key in epoch_metrics:
                epoch_metrics[key] /= folds_count
            
            # Store average metrics
            all_metrics['train_loss'].append(epoch_metrics['train_loss'])
            all_metrics['val_loss'].append(epoch_metrics['val_loss'])
            all_metrics['train_mse'].append(epoch_metrics['train_mse'])
            all_metrics['val_mse'].append(epoch_metrics['val_mse'])
            all_metrics['train_pearson'].append(epoch_metrics['train_pearson'])
            all_metrics['val_pearson'].append(epoch_metrics['val_pearson'])
            all_metrics['epoch'].append(epoch)
            all_metrics['timestamp'].append(datetime.now().isoformat())
        
        # Save the average metrics
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(self.experiment_dir / 'avg_metrics.csv', index=False)
        
        # Save the fold-specific metrics
        fold_metrics_df = pd.DataFrame(fold_specific_metrics)
        fold_metrics_df.to_csv(self.experiment_dir / 'fold_metrics.csv', index=False)
        
        # Add experiment results to the tracker
        updated_results = self.tracker.add_experiment_result(
            self.experiment_id, 
            self.config, 
            metrics_df
        )
        
        # Print out the updated results table for easy reference
        print("\n=== Updated Experiment Results ===")
        summary_columns = [
            'experiment_id', 
            'glycan_encoder_type', 
            'protein_encoder_type', 
            'binding_predictor_type',
            'best_train_loss', 
            'best_val_loss',
            'best_train_mse',
            'best_val_mse',
            'best_train_pearson',
            'best_val_pearson',
        ]
        print(updated_results[summary_columns].to_string())
        
        # Save a copy of the results in this experiment's directory for reference
        updated_results.to_csv(self.experiment_dir / 'experiment_results.csv', index=False)
        updated_results[summary_columns].to_csv('experiments_summary.csv', index=False)
        
        # Plot the metrics
        self.plot_metrics(metrics_df, fold_metrics_df)
        
        # Train a final model on all data if needed
        if self.config.train_final_model:
            print("\n" + "="*50)
            print("Training final model on all data")
            
            # Reset the models
            self.reset_models()
            
            full_train_pytorch_dataset = GlycoProteinDataset(
                fractions_df, glycan_encodings, protein_encodings, glycan_mapping, protein_mapping
            )
            
            # Create a data loader for all data
            full_loader = DataLoader(
                full_train_pytorch_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            # Train for the specified number of epochs
            for epoch in range(self.config.num_epochs):
                print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
                
                # Use default weight for final model training
                train_metrics = self._train_epoch(full_loader, 1.0)
                
                print(f"Train Loss: {train_metrics['loss']:.4f}")
                
                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    self.save_checkpoint(epoch=epoch+1)
            
            # Save the final model
            self.save_checkpoint()