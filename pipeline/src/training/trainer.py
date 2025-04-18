import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
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
            'loss_type': config.loss_type,
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



class LossesClass(nn.Module):
    def __init__(self, loss_type='mse', delta=1.0, beta=1.0):
        """
        Combined loss function that supports multiple loss types
        
        Args:
            loss_type: One of 'mse', 'rmse', 'rmsle', 'mae', 'log_mae', 'huber', 'smooth_l1'
            delta: Parameter for Huber loss
            beta: Parameter for Smooth L1 loss
        """
        super().__init__()
        
        self.loss_type = loss_type.lower()
        self.delta = delta
        self.beta = beta
    
    
    def forward(self, inputs, targets, weights):
        # Check if all weights are approximately 1.0
        all_ones = weights == 1.0
        
        # Handle loss types that can use weights directly
        if self.loss_type == 'mse':
            loss = (inputs - targets)**2
            return torch.mean(loss * weights)
        
        elif self.loss_type == 'rmse':
            loss = (inputs - targets)**2
            weighted_mean = torch.mean(loss * weights)
            return torch.sqrt(weighted_mean)
        
        elif self.loss_type == 'rmsle':
            log_inputs = torch.log(inputs + 1)
            log_targets = torch.log(targets + 1)
            loss = (log_inputs - log_targets)**2
            weighted_mean = torch.mean(loss * weights)
            return torch.sqrt(weighted_mean)
        
        elif self.loss_type == 'mae' or self.loss_type == 'l1':
            loss = torch.abs(inputs - targets)
            return torch.mean(loss * weights)
        
        elif self.loss_type == 'log_mae' or self.loss_type == 'log_l1':
            loss = torch.abs(torch.log(inputs + 1) - torch.log(targets + 1))
            return torch.mean(loss * weights)
        
        # For loss types that don't directly use weights, use PyTorch's implementation
        # and effectively ignore weights when they're all 1.0
        
        elif self.loss_type == 'huber':
            # If weights are all approximately 1.0, use standard implementation
            if all_ones:
                return F.huber_loss(inputs, targets, reduction='mean', delta=self.delta)
            else:
                # Only apply manual weighting if weights are not all 1.0
                loss = F.huber_loss(inputs, targets, reduction='none', delta=self.delta)
                return torch.mean(loss * weights)
        
        elif self.loss_type == 'smooth_l1':
            # If weights are all approximately 1.0, use standard implementation
            if all_ones:
                return F.smooth_l1_loss(inputs, targets, reduction='mean', beta=self.beta)
            else:
                # Only apply manual weighting if weights are not all 1.0
                loss = F.smooth_l1_loss(inputs, targets, reduction='none', beta=self.beta)
                return torch.mean(loss * weights)
        
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")



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
        
        self.criterion = LossesClass(loss_type=self.config.loss_type, delta=self.config.delta, beta=self.config.beta)
    
    
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
            predictions = self.binding_predictor(
                glycan_encoding,
                protein_encoding,
                concentration
            )

            # Prevent double-log when using both log_predict=True and a log-based loss
            apply_log = self.config.log_predict and self.config.loss_type not in ['rmsle', 'log_mae', 'log_l1']
            if apply_log:
                targets = torch.log(targets + 1)
                predictions = torch.log(predictions + 1)
            
            
            loss = self.criterion(predictions, targets, fold_weight)
            
            # reset gradients to zero
            self.optimizer.zero_grad()
            # perform backpropagation to calculate the gradients we need to improve model
            loss.backward(retain_graph=True)
            # update the weights with our loss gradients
            self.optimizer.step()
            
            # revert predictions and targets to original values for original analysis if using log transform
            if apply_log:
                predictions = torch.exp(predictions) - 1
                targets = torch.exp(targets) - 1
            
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
        metrics['loss'] = total_loss
        
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
                predictions = self.binding_predictor(
                    glycan_encoding,
                    protein_encoding,
                    concentration
                )

                # Prevent double-log when using both log_predict=True and a log-based loss
                apply_log = self.config.log_predict and self.config.loss_type not in ['rmsle', 'log_mae', 'log_l1']
                if apply_log:
                    targets = torch.log(targets + 1)
                    predictions = torch.log(predictions + 1)
                
                loss = self.criterion(predictions, targets, fold_weight)
                
                # revert predictions and targets to original values for original analysis if using log transform
                if apply_log:
                    predictions = torch.exp(predictions) - 1
                    targets = torch.exp(targets) - 1
                
                # track totals
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(targets)
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # save metrics
        val_predictions = torch.cat(all_predictions)
        val_targets = torch.cat(all_targets)
        
        # Print out the single highest prediction
        max_prediction = torch.max(val_predictions).item()
        print(f"Highest prediction value: {max_prediction:.6f}")
        
        metrics = calculate_metrics(val_predictions, val_targets)
        metrics['loss'] = total_loss
        
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
        raw_errors = predictions_np - targets_np
        
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
    
    
    def create_scatter_plot(self, predictions: torch.Tensor, targets: torch.Tensor, prefix: str = "scatter"):
        """
        Create a scatterplot of true binding values vs predicted binding values.
        
        Args:
            predictions: Tensor of predicted binding values.
            targets: Tensor of true binding values.
            prefix: Optional prefix for the filename.
        """
        # Convert tensors to numpy arrays and flatten for plotting
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        
        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(targets_np, predictions_np, alpha=0.6, color='skyblue', edgecolor='k', label='Predictions')
        
        # Create a reference line y = x (ideal prediction line)
        min_val = min(targets_np.min(), predictions_np.min())
        max_val = max(targets_np.max(), predictions_np.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal')
        
        # Customize labels and title
        plt.xlabel('True Binding Value')
        plt.ylabel('Predicted Binding Value')
        plt.title(f'{prefix} Scatterplot: True vs Predicted Binding Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ensure the plots directory exists and save the plot
        plots_dir = self.experiment_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / f'{prefix}_scatterplot.png')
        plt.close()
    
    
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
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = samples_df['f'].values
        absolute_errors = np.abs(predictions_np - targets_np)
        worst_df = samples_df.copy()
        worst_df['Prediction'] = predictions_np
        worst_df['Target'] = targets_np
        worst_df['Absolute_Error'] = absolute_errors
        worst_df = worst_df.sort_values(by='Absolute_Error', ascending=False).head(num_samples)
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
            self.config.glycan_type,
            self.config.random_state,
            self.config.split_mode,
            self.config.use_kfold,
            self.config.k_folds,
            self.config.val_split,
            torch.device(self.config.device)
        )
        
        # Create mappings
        glycan_mapping = {name: idx for idx, name in enumerate(glycans_df['Name'])}
        protein_mapping = {name: idx for idx, name in enumerate(proteins_df['ProteinGroup'])}
        
        # Tracking metrics
        fold_metrics = []
        fold_specific_metrics = []
        
        if self.config.use_kfold:
            print(f"Starting {self.config.k_folds}-fold cross-validation on our total {len(fractions_df)} samples")
            train_sizes = [len(train_samples) for train_samples, test_samples in fold_indices]
            total_train_samples = sum(train_sizes)
            fold_weights = [total_train_samples / (len(fold_indices) * train_size) for train_size in train_sizes]
        else:
            print(f"Starting regular training with {len(fold_indices[0][0])} training samples and {len(fold_indices[0][1])} validation samples")
            fold_weights = [1.0]
        
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
                    # Comment out inference analysis plots for now (having matplotlib issues)
                    """
                    error_stats = self.create_error_histogram(
                        val_predictions, val_targets, 
                        prefix=f"fold{fold_idx}_final",
                        train_predictions=train_predictions, 
                        train_targets=train_targets
                    )
                    
                    
                    self.create_scatter_plot(val_predictions, val_targets, prefix=f"fold{fold_idx}_final")
                    
                    # Save the worst predictions from the validation set (uses the original val_data DataFrame)
                    self.save_worst_predictions(val_data, val_predictions, val_targets)
                    
                    self.create_filtered_error_histogram(
                        val_predictions, val_targets, val_data,
                        target_threshold=0.2,
                        prefix=f"fold{fold_idx}_high_targets_0.2",
                        train_predictions=train_predictions,
                        train_targets=train_targets,
                        train_samples_df=train_data
                    )
                    
                    # For targets > 0.5
                    self.create_filtered_error_histogram(
                        val_predictions, val_targets, val_data,
                        target_threshold=0.5,
                        prefix=f"fold{fold_idx}_high_targets_0.5",
                        train_predictions=train_predictions,
                        train_targets=train_targets,
                        train_samples_df=train_data
                    )
                    
                    # Save best predictions for high-value targets
                    self.save_best_predictions_by_threshold(
                        val_data, val_predictions, 
                        thresholds=[0.2, 0.5], 
                        num_samples=10
                    )
                    """
                
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
                
                print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                
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
            'loss_type',
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
        
        if self.config.eval_testset:
            self.testset_eval(glycan_encodings, protein_encodings, glycan_mapping, protein_mapping)
        
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
                train_metrics, epoch_predictions, epoch_targets = self._train_epoch(full_loader, 1.0)
                
                print(f"Train Loss: {train_metrics['loss']:.4f}")
                
                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    self.save_checkpoint(epoch=epoch+1)
            
                    
            if self.config.eval_testset:
                self.testset_eval(glycan_encodings, protein_encodings, glycan_mapping, protein_mapping)
                
            # Save the final model
            self.save_checkpoint()
    
    def testset_eval(self, glycan_encodings, protein_encodings, glycan_mapping, protein_mapping):
        # This is a hacky fix to run on testset cause for some reason run_testset.py doesnt work and dont have time to fix it. 
        # This is bad impolementation right after training cause it encourages adjuting training on test set results but I warned the team about this already so we should be good.
        test_fractions_df = pd.read_csv(self.config.testdata_path, sep='\t') #'data/Test_Fractions.csv', sep='\t')        
        test_dataset = GlycoProteinDataset(
            test_fractions_df, glycan_encodings, protein_encodings, glycan_mapping, protein_mapping
        )
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_metrics, val_predictions, val_targets = self._validate(test_loader, 1.0)
        
        print('TEST metrics', test_metrics)
    
    def create_filtered_error_histogram(self, predictions, targets, samples_df, target_threshold=0.5, 
                               prefix="filtered", train_predictions=None, train_targets=None, 
                               train_samples_df=None):
        """
        Create histograms of prediction errors for samples where the target is above a threshold.
        
        Args:
            predictions: Tensor of predictions (validation set)
            targets: Tensor of target values (validation set)
            samples_df: DataFrame with sample data corresponding to predictions/targets
            target_threshold: Filter for targets above this value (default 0.5)
            prefix: Optional prefix for the filename (e.g., 'high_targets')
            train_predictions: Optional tensor of training predictions for comparison
            train_targets: Optional tensor of training targets for comparison
            train_samples_df: DataFrame with sample data for training set
        """
        prefix = f"{self.config.protein_encoder_type}_{self.config.glycan_encoder_type}_{self.config.binding_predictor_type}_{prefix}"
        
        # Convert targets to numpy to match samples_df shape
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = samples_df['f'].values
        
        # Create filter mask for high-value targets
        high_targets_mask = targets_np > target_threshold
        
        # Apply the filter to get only high-value targets and their predictions
        filtered_predictions = predictions_np[high_targets_mask]
        filtered_targets = targets_np[high_targets_mask]
        
        # Skip if there are no samples above the threshold
        if len(filtered_targets) == 0:
            print(f"No samples with targets above {target_threshold} found.")
            return None
        
        # Calculate errors (absolute and raw)
        absolute_errors = np.abs(filtered_predictions - filtered_targets)
        raw_errors = filtered_predictions - filtered_targets
        
        # Process training data if provided
        has_training_data = (train_predictions is not None and train_targets is not None 
                            and train_samples_df is not None)
        if has_training_data:
            train_predictions_np = train_predictions.cpu().numpy().flatten()
            train_targets_np = train_samples_df['f'].values
            # Filter training data as well
            train_high_targets_mask = train_targets_np > target_threshold
            train_filtered_predictions = train_predictions_np[train_high_targets_mask]
            train_filtered_targets = train_targets_np[train_high_targets_mask]
            # Calculate errors for filtered training data
            train_absolute_errors = np.abs(train_filtered_predictions - train_filtered_targets)
            train_raw_errors = train_filtered_predictions - train_filtered_targets
        
        bins = 30
        
        # Save dir
        plots_dir = self.experiment_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Absolute Error Histogram (Stacked, no best fit line)
        plt.figure(figsize=(10, 6))
        
        max_error = absolute_errors.max() if len(absolute_errors) > 0 else 1.0
        if has_training_data and len(train_absolute_errors) > 0:
            max_error = max(max_error, train_absolute_errors.max())
        
        hist_bins = np.linspace(0, max_error * 1.05, bins)
        
        if has_training_data and len(train_filtered_targets) > 0 and len(filtered_targets) > 0:
            plt.hist([absolute_errors, train_absolute_errors], bins=hist_bins, 
                     label=['Validation', 'Training'], alpha=0.7, 
                     color=['skyblue', 'teal'], edgecolor=['navy', 'darkgreen'],
                     stacked=True)
        elif len(filtered_targets) > 0:
            plt.hist(absolute_errors, bins=hist_bins, alpha=0.7, 
                     color='skyblue', edgecolor='navy', label='Validation')
        
        plt.xlabel('Absolute Error')
        plt.ylabel('Count')
        plt.title(f'{prefix} Absolute Error Distribution (Targets > {target_threshold})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if len(filtered_targets) > 0:
            mean_error = np.mean(absolute_errors)
            median_error = np.median(absolute_errors)
            
            if has_training_data and len(train_filtered_targets) > 0:
                train_mean_error = np.mean(train_absolute_errors)
                train_median_error = np.median(train_absolute_errors)
                combined_abs_errors = np.concatenate([absolute_errors, train_absolute_errors])
                combined_mean = np.mean(combined_abs_errors)
                combined_median = np.median(combined_abs_errors)
                stats_text = f'Combined Mean: {combined_mean:.4f}\nCombined Median: {combined_median:.4f}'
                stats_text += f'\nSamples: Val={len(filtered_targets)}, Train={len(train_filtered_targets)}'
            else:
                stats_text = f'Mean: {mean_error:.4f}\nMedian: {median_error:.4f}'
                stats_text += f'\nSamples: {len(filtered_targets)}'
            
            plt.annotate(stats_text, xy=(0.65, 0.75), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.savefig(plots_dir / f'{prefix}_absolute_error_threshold_{target_threshold}.png')
        plt.close()
        
        # 2. Raw Error Distribution (Stacked with negative values and best fit line)
        plt.figure(figsize=(10, 6))
        
        if len(filtered_targets) > 0:
            min_raw_error = raw_errors.min()
            max_raw_error = raw_errors.max()
            
            if has_training_data and len(train_filtered_targets) > 0:
                min_raw_error = min(min_raw_error, train_raw_errors.min())
                max_raw_error = max(max_raw_error, train_raw_errors.max())
            
            raw_hist_bins = np.linspace(min_raw_error * 1.05, max_raw_error * 1.05, bins)
            
            if has_training_data and len(train_filtered_targets) > 0:
                plt.hist([raw_errors, train_raw_errors], bins=raw_hist_bins, 
                         label=['Validation', 'Training'], alpha=0.7, 
                         color=['skyblue', 'teal'], edgecolor=['navy', 'darkgreen'],
                         stacked=True)
                
                from scipy.stats import gaussian_kde
                x_range = np.linspace(min_raw_error * 1.05, max_raw_error * 1.05, 1000)
                combined_raw_errors = np.concatenate([raw_errors, train_raw_errors])
                
                if len(combined_raw_errors) > 1:
                    combined_kde = gaussian_kde(combined_raw_errors)
                    plt.plot(x_range, combined_kde(x_range) * len(combined_raw_errors) * (raw_hist_bins[1] - raw_hist_bins[0]), 
                             'r-', linewidth=2, label='Error Distribution')
            else:
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
            plt.title(f'{prefix} Error Distribution (Targets > {target_threshold})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            if has_training_data and len(train_filtered_targets) > 0:
                mean_raw_error = np.mean(combined_raw_errors)
                std_raw_error = np.std(combined_raw_errors)
            else:
                mean_raw_error = np.mean(raw_errors)
                std_raw_error = np.std(raw_errors)
            
            stats_text = f'Mean: {mean_raw_error:.4f}\nStd Dev: {std_raw_error:.4f}'
            plt.annotate(stats_text, xy=(0.05, 0.75), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.savefig(plots_dir / f'{prefix}_error_distribution_threshold_{target_threshold}.png')
        plt.close()
        
        if len(filtered_targets) > 0:
            result = {
                'mean_abs_error': mean_error if not has_training_data or len(train_filtered_targets) == 0 else combined_mean,
                'median_abs_error': median_error if not has_training_data or len(train_filtered_targets) == 0 else combined_median,
                'mean_raw_error': mean_raw_error,
                'std_raw_error': std_raw_error,
                'validation_samples': len(filtered_targets),
                'training_samples': len(train_filtered_targets) if has_training_data else 0,
                'threshold': target_threshold
            }
            return result
        else:
            return {
                'validation_samples': 0,
                'training_samples': 0,
                'threshold': target_threshold
            }
    
    
    def save_best_predictions_by_threshold(self, samples_df: pd.DataFrame, predictions: torch.Tensor, 
                                           thresholds: list = [0.2, 0.5], num_samples: int = 10):
        """
        Save the best predictions (based on lowest absolute error) for samples with targets above specified thresholds.
        
        Args:
            samples_df: DataFrame corresponding to the validation samples
            predictions: Tensor of predictions (from validation)
            thresholds: List of target thresholds to filter by (default [0.2, 0.5])
            num_samples: Number of best samples to save for each threshold (default is 10)
        """
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = samples_df['f'].values
        absolute_errors = np.abs(predictions_np - targets_np)
        results_df = samples_df.copy()
        results_df['Prediction'] = predictions_np
        results_df['Target'] = targets_np
        results_df['Absolute_Error'] = absolute_errors
        output_dir = self.experiment_dir / 'predictions'
        output_dir.mkdir(exist_ok=True)
        for threshold in thresholds:
            filtered_df = results_df[results_df['Target'] > threshold].copy()
            if len(filtered_df) == 0:
                print(f"No samples with targets above {threshold} found.")
                continue
            best_predictions = filtered_df.sort_values(by='Absolute_Error').head(num_samples)
            best_predictions['Relative_Error'] = (best_predictions['Absolute_Error'] / best_predictions['Target']) * 100
            stats_df = pd.DataFrame({
                'ProteinGroup': [f'Total samples > {threshold}:'],
                'GlycanID': [len(filtered_df)],
                'Target': [f'Mean Abs Error: {filtered_df["Absolute_Error"].mean():.4f}'],
                'Prediction': [f'Mean Rel Error: {(filtered_df["Absolute_Error"] / filtered_df["Target"] * 100).mean():.2f}%'],
                'Absolute_Error': [f'Median Abs Error: {filtered_df["Absolute_Error"].median():.4f}'],
                'Relative_Error': [f'Median Rel Error: {(filtered_df["Absolute_Error"] / filtered_df["Target"] * 100).median():.2f}%']
            })
            output_file = output_dir / f'best_predictions_threshold_{threshold}.csv'
            pd.concat([best_predictions, stats_df], ignore_index=True).to_csv(output_file, index=False)
            print(f"Saved best {num_samples} predictions for targets > {threshold} (from {len(filtered_df)} samples)")
            
            worst_predictions = filtered_df.sort_values(by='Absolute_Error', ascending=False).head(num_samples)
            worst_predictions['Relative_Error'] = (worst_predictions['Absolute_Error'] / worst_predictions['Target']) * 100
            output_file = output_dir / f'worst_predictions_threshold_{threshold}.csv'
            pd.concat([worst_predictions, stats_df], ignore_index=True).to_csv(output_file, index=False)
 