# References:
# https://github.com/victoresque/pytorch-template/blob/master/base/base_trainer.py
# Used GitHub Copilot to generate parts of this script (I provided the pytorch base_trainer as a prompt)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
from tqdm import tqdm

from ..utils.config import TrainingConfig
from ..utils.metrics import calculate_metrics
from ..utils.model_factory import create_binding_predictor, create_glycan_encoder, create_protein_encoder
from ..data.dataset import prepare_datasets

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
            'protein_dim': 52
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
    
    def save_checkpoint(self, epoch: int = None):
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
        checkpoint_name = f"model_epoch_{epoch}.pt" if epoch else "model_final.pt"
        torch.save(
            checkpoint,
            checkpoint_dir / checkpoint_name
        )
    
    def plot_metrics(self, metrics_df: pd.DataFrame):
        plots_dir = self.experiment_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
        plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'loss_curves.png')
        plt.close()
        
        # plot MSE curves
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epoch'], metrics_df['train_mse'], label='Train MSE')
        plt.plot(metrics_df['epoch'], metrics_df['val_mse'], label='Validation MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Training and Validation MSE')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'mse_curves.png')
        plt.close()
        
        # plot Pearson correlation curves
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epoch'], metrics_df['train_pearson'], label='Train Pearson')
        plt.plot(metrics_df['epoch'], metrics_df['val_pearson'], label='Validation Pearson')
        plt.xlabel('Epoch')
        plt.ylabel('Pearson Correlation')
        plt.title('Training and Validation Pearson Correlation')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'pearson_curves.png')
        plt.close()
    
    def train(self, data_df: pd.DataFrame, precomputed_path: str = None):
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
        
        train_dataset, val_dataset = prepare_datasets(
            data_df,
            self.config.val_split,
            self.glycan_encoder,
#            self.protein_encoder,
            precomputed_path
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        print(f"Training with {len(train_dataset)} samples, "
              f"validating with {len(val_dataset)} samples")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            
            # save metrics
            timestamp = datetime.now().isoformat()
            all_metrics['train_loss'].append(train_metrics['loss'])
            all_metrics['val_loss'].append(val_metrics['loss'])
            all_metrics['train_mse'].append(train_metrics['mse'])
            all_metrics['val_mse'].append(val_metrics['mse'])
            all_metrics['train_pearson'].append(train_metrics['pearson'])
            all_metrics['val_pearson'].append(val_metrics['pearson'])
            all_metrics['epoch'].append(epoch)
            all_metrics['timestamp'].append(timestamp)
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}")
            
            if (epoch + 1) % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(epoch=epoch)
        
        # save the final model
        self.save_checkpoint()
        
        # save the final metrics
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(self.experiment_dir / 'metrics.csv', index=False)
        self.plot_metrics(metrics_df)