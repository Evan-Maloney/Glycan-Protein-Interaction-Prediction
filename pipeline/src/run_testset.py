import torch
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path
import argparse
import json
#from data.dataset import GlycoProteinDataset
from torch.utils.data import DataLoader
from utils.model_factory import create_binding_predictor, create_glycan_encoder, create_protein_encoder
from utils.config import TrainingConfig

from training.trainer import GlycoProteinDataset

# Metrics calculation function
def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate training/validation metrics
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): True values
        
    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    # convert values to numpy arrays
    preds_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    mse = np.mean((preds_np - targets_np) ** 2)
    pearson_corr, _ = pearsonr(preds_np.flatten(), targets_np.flatten())
    
    return {
        'mse': float(mse),
        'pearson': float(pearson_corr)
    }

class ModelLoader:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.config = None
        self.model = None
        
    def load_config(self):
        """Load the model configuration from checkpoint"""
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.config = checkpoint['config']
        return {
            'glycan_encoder_type': self.config.glycan_encoder_type,
            'protein_encoder_type': self.config.protein_encoder_type,
            'binding_predictor_type': self.config.binding_predictor_type,
            'glycan_type': self.config.glycan_type,
            'glycans_data_path': self.config.glycans_data_path,
            'proteins_data_path': self.config.proteins_data_path,
            'predict_data_path': 'data/Test_Fractions.csv'  # Set to the test data path
        }
    
    def load_model(self):
        """Load the model weights from checkpoint"""
        
        if self.config is None:
            self.load_config()
        
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        
        # Create models
        self.glycan_encoder = create_glycan_encoder(self.config.glycan_encoder_type)
        self.protein_encoder = create_protein_encoder(self.config.protein_encoder_type)
        
        # Build binding predictor
        predictor_params = {
            'glycan_dim': self.glycan_encoder.embedding_dim,
            'protein_dim': self.protein_encoder.embedding_dim
        }
        
        if 'dnn_hidden_dims' in self.config.model_specific_params:
            predictor_params['hidden_dims'] = self.config.model_specific_params['dnn_hidden_dims']
        
        self.binding_predictor = create_binding_predictor(
            self.config.binding_predictor_type,
            **predictor_params
        )
        
        # Load weights
        self.glycan_encoder.load_state_dict(checkpoint['glycan_encoder'])
        self.protein_encoder.load_state_dict(checkpoint['protein_encoder'])
        self.binding_predictor.load_state_dict(checkpoint['binding_predictor'])
        
        return self.glycan_encoder, self.protein_encoder, self.binding_predictor
    
    def predict(self, fractions_df, glycans_df, proteins_df):
        """Make predictions using the loaded model"""
        
        if not hasattr(self, 'binding_predictor'):
            self.load_model()
        
        # Create mappings and encodings
        glycan_mapping = {name: idx for idx, name in enumerate(glycans_df['Name'])}
        protein_mapping = {name: idx for idx, name in enumerate(proteins_df['ProteinGroup'])}
        
        # Get encodings
        glycan_encodings = []
        for _, glycan in glycans_df.iterrows():
            glycan_encodings.append(self.glycan_encoder.encode(glycan))
        glycan_encodings = torch.stack(glycan_encodings)
        
        protein_encodings = []
        for _, protein in proteins_df.iterrows():
            protein_encodings.append(self.protein_encoder.encode(protein))
        protein_encodings = torch.stack(protein_encodings)
        
        # Create dataset and dataloader
        test_dataset = GlycoProteinDataset(
            fractions_df, glycan_encodings, protein_encodings, glycan_mapping, protein_mapping
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Make predictions
        self.glycan_encoder.eval()
        self.protein_encoder.eval()
        self.binding_predictor.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                glycan_encoding = batch['glycan_encoding']
                protein_encoding = batch['protein_encoding']
                concentration = batch['concentration']
                targets = batch['target']
                
                predictions = self.binding_predictor(glycan_encoding, protein_encoding, concentration)
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Calculate metrics
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        metrics = calculate_metrics(predictions, targets)
        
        return predictions, targets, metrics


def load_and_predict(checkpoint_path, test_data_path):
    """Load model and make predictions on test data"""
    loader = ModelLoader(checkpoint_path)
    config_info = loader.load_config()
    
    # Load test data
    fractions_df = pd.read_csv(test_data_path)
    glycans_df = pd.read_csv(config_info['glycans_data_path'])
    proteins_df = pd.read_csv(config_info['proteins_data_path'])
    
    # Make predictions
    predictions, targets, metrics = loader.predict(fractions_df, glycans_df, proteins_df)
    
    return {
        'config': config_info,
        'metrics': metrics,
        'predictions': predictions.numpy(),
        'targets': targets.numpy()
    }
    
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    
    config = TrainingConfig.from_yaml(args.config)
    
    
if __name__ == "__main__":
    main()