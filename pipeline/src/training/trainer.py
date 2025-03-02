# References:
# https://github.com/victoresque/pytorch-template/blob/master/base/base_trainer.py
# Used GitHub Copilot to generate parts of this script (I provided the pytorch base_trainer as a prompt)
# Used Claude 3.7 to update to cross val

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
from tqdm import tqdm

from ..utils.config import TrainingConfig
from ..utils.metrics import calculate_metrics
from ..utils.model_factory import create_binding_predictor, create_glycan_encoder, create_protein_encoder
from ..data.dataset import prepare_kfold_datasets

class FractionDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get row by position, not by index
        row = self.data.iloc[idx]
        
        # Make sure all tensors are proper PyTorch tensors
        glycan_encoding = torch.tensor(row['glycan_encoding'], dtype=torch.float32)
        protein_encoding = torch.tensor(row['protein_encoding'], dtype=torch.float32)
        
        # Make sure scalars are proper Python types
        concentration = float(row['concentration'])
        target = float(row['target'])
        
        return {
            'glycan_encoding': glycan_encoding,
            'protein_encoding': protein_encoding,
            'concentration': torch.tensor([concentration], dtype=torch.float32),
            'target': torch.tensor([target], dtype=torch.float32)
        }
        
def custom_collate(batch):
    # Filter out items with invalid tensors
    valid_items = []
    for item in batch:
        # Check if tensors are valid (have dimensions)
        if (hasattr(item['glycan_encoding'], 'shape') and 
            hasattr(item['protein_encoding'], 'shape') and
            item['glycan_encoding'].dim() > 0 and 
            item['protein_encoding'].dim() > 0):
            valid_items.append(item)
        else:
            print(f"Skipping invalid item. Glycan: {item['glycan_encoding']}, Protein: {item['protein_encoding']}")
    
    if not valid_items:
        print("Warning: Entire batch contained invalid items")
        # Return a dummy batch or skip this batch
        return None
    
    # Stack the valid items
    try:
        glycan_encodings = torch.stack([item['glycan_encoding'] for item in valid_items])
        protein_encodings = torch.stack([item['protein_encoding'] for item in valid_items])
        concentrations = torch.stack([item['concentration'] for item in valid_items])
        targets = torch.stack([item['target'] for item in valid_items])
        
        return {
            'glycan_encoding': glycan_encodings,
            'protein_encoding': protein_encodings,
            'concentration': concentrations,
            'target': targets
        }
    except Exception as e:
        print(f"Error during collation: {e}")
        # Print shapes for debugging
        for i, item in enumerate(valid_items):
            print(f"Item {i}: glycan={item['glycan_encoding'].shape}, protein={item['protein_encoding'].shape}")
        return None

class BindingTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.experiment_dir = Path(config.output_dir)
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
        self.criterion = nn.MSELoss()
        
    def reset_models(self):
        """Reset all models to their initial state"""
        self.setup_models()
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
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
            # add all features here
            
            targets = batch['target'].to(self.device)
            
            predictions = self.binding_predictor(
                glycan_encoding,
                protein_encoding,
                concentration
            )
            
            loss = self.criterion(predictions, targets)
            
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
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
        
        return metrics
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
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

                loss = self.criterion(predictions, targets)
                
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
        
        return metrics
    
    
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
        else:
            checkpoint_name = "model_final.pt"
            
        torch.save(
            checkpoint,
            checkpoint_dir / checkpoint_name
        )
    
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
    
    def train(self, fractions_df: pd.DataFrame, glycans_df: pd.DataFrame, proteins_df: pd.DataFrame):
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
        
        fold_indices, full_fractions_df = prepare_kfold_datasets(
            fractions_df,
            glycans_df,
            proteins_df,
            self.config.k_folds,
            self.glycan_encoder,
            self.protein_encoder,
            self.config.random_state
        )
        
        
        # Set up metrics tracking
        fold_metrics = []
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
        
        # Tracking metrics per fold for visualization
        fold_specific_metrics = []
        
        print(f"Starting {self.config.k_folds}-fold cross-validation with {len(full_fractions_df)} samples")
        
        # For each fold
        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
            # Get data for this fold
            train_data = full_fractions_df.loc[train_idx]
            val_data = full_fractions_df.loc[test_idx]
            
            
            print(f"\n{'='*20} Fold {fold_idx+1}/{self.config.k_folds} {'='*20}")
            
            # Reset models for this fold
            self.reset_models()
            

            
            # Create data loaders
            train_loader = DataLoader(
                FractionDataset(train_data),
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=custom_collate
            )
            val_loader = DataLoader(
                FractionDataset(val_data),
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=custom_collate
            )
            
            print(f"Training with {len(train_data)} samples, "
                  f"validating with {len(val_data)} samples")
            
            # Training loop for this fold
            fold_train_metrics = []
            fold_val_metrics = []
            
            for epoch in range(self.config.num_epochs):
                print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
                
                train_metrics = self._train_epoch(train_loader)
                val_metrics = self._validate(val_loader)
                
                # Save metrics for this fold
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
                    self.save_checkpoint(fold=fold_idx, epoch=epoch)
            
            # Save final model for this fold
            self.save_checkpoint(fold=fold_idx)
            
            # Record metrics from this fold
            fold_metrics.append({
                'fold': fold_idx,
                'train_metrics': fold_train_metrics,
                'val_metrics': fold_val_metrics
            })
        
        # Calculate average metrics across all folds
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
            for key in epoch_metrics:
                epoch_metrics[key] /= self.config.k_folds
            
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
        
        # Plot the metrics
        self.plot_metrics(metrics_df, fold_metrics_df)
        
        # Train a final model on all data if needed
        if self.config.train_final_model:
            print("\n" + "="*50)
            print("Training final model on all data")
            
            # Reset the models
            self.reset_models()
            
            # Create a data loader for all data
            full_loader = DataLoader(
                full_fractions_df,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            # Train for the specified number of epochs
            for epoch in range(self.config.num_epochs):
                print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
                
                train_metrics = self._train_epoch(full_loader)
                
                print(f"Train Loss: {train_metrics['loss']:.4f}")
                
                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    self.save_checkpoint(epoch=epoch)
            
            # Save the final model
            self.save_checkpoint()