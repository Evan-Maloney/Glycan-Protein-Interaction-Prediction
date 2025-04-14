import torch
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import argparse
from tqdm import tqdm
#from data.dataset import GlycoProteinDataset
from torch.utils.data import DataLoader
from src.utils.model_factory import create_binding_predictor, create_glycan_encoder, create_protein_encoder
from src.utils.config import TrainingConfig

from src.training.trainer import GlycoProteinDataset

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
    """
    if np.std(preds_np) == 0 or np.std(targets_np) == 0:
        pearson_corr = float('nan')  # Explicitly set to nan if constant
        
        # Add diagnostics
        print(f"Predictions std: {np.std(preds_np)}")
        print(f"Targets std: {np.std(targets_np)}")
        print(f"Predictions: {preds_np[:5]}...")  # Show first few values
        print(f"Targets: {targets_np[:5]}...")    # Show first few values
    else:
    """
    pearson_corr, _ = pearsonr(preds_np.flatten(), targets_np.flatten())
    
    return {
        'mse': float(mse),
        'pearson': float(pearson_corr)
    }
    
def batch_encode(encoder, data_list, device, batch_size):
    """Process data in batches to avoid CUDA memory overflow"""
    all_encodings = []
    total_items = len(data_list)
    
    for i in range(0, total_items, batch_size):
        # Get current batch
        batch = data_list[i:min(i+batch_size, total_items)]
        
        # Encode batch
        batch_encodings = encoder.encode_batch(batch, device)
        all_encodings.append(batch_encodings)
        
        # Print progress
        print(f'Progress: {min(i+batch_size, total_items)}/{total_items}')
        
        # Optional: clear CUDA cache to prevent memory fragmentation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    return torch.cat(all_encodings, dim=0)

class ModelLoader:
    def __init__(self, checkpoint_path, DEVICE):
        #self.checkpoint_path = checkpoint_path
 
        
        self.device = DEVICE
        
        try:
            self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.config = self.checkpoint['config']
        except Exception as e:
            print('Error loading model weights:\n')
            raise e
        
    def load_model(self):
        """Load the model weights from checkpoint"""
        
        
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
        self.glycan_encoder.load_state_dict(self.checkpoint['glycan_encoder'])
        self.protein_encoder.load_state_dict(self.checkpoint['protein_encoder'])
        self.binding_predictor.load_state_dict(self.checkpoint['binding_predictor'])
        
        return self.glycan_encoder, self.protein_encoder, self.binding_predictor
    
    def predict(self, fractions_df, glycans_df, proteins_df):
        """Make predictions using the loaded model"""
        
        if not hasattr(self, 'binding_predictor'):
            self.load_model()
        
        # Create mappings and encodings
        glycan_mapping = {name: idx for idx, name in enumerate(glycans_df['Name'])}
        protein_mapping = {name: idx for idx, name in enumerate(proteins_df['ProteinGroup'])}
        
        # Get encodings  
         # only do batch to not overload RAM of GPU
        if self.device.type == 'cuda':
            batch_size = 100  # Adjust based on your GPU memory

            # Encode glycans in batches
            glycan_encodings = batch_encode(
                self.glycan_encoder, 
                glycans_df[self.config.glycan_type].tolist(), 
                self.device, 
                batch_size=batch_size
            )

            # Encode proteins in batches
            protein_encodings = batch_encode(
                self.protein_encoder, 
                self.proteins_df['Amino Acid Sequence'].tolist(), 
                self.device, 
                batch_size=batch_size
            )
        else:
            glycan_encodings = self.glycan_encoder.encode_batch(glycans_df[self.config.glycan_type].tolist(), self.device)
            protein_encodings = self.protein_encoder.encode_batch(proteins_df['Amino Acid Sequence'].tolist(), self.device)
        
        
        # Create dataset and dataloader
        test_dataset = GlycoProteinDataset(
            fractions_df, glycan_encodings, protein_encodings, glycan_mapping, protein_mapping
        )
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Make predictions
        self.glycan_encoder.eval()
        self.protein_encoder.eval()
        self.binding_predictor.eval()
        
        all_predictions = []
        all_targets = []
        
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Predicting')
            for batch in pbar:
                glycan_encoding = batch['glycan_encoding'].to(self.device)
                protein_encoding = batch['protein_encoding'].to(self.device)
                concentration = batch['concentration'].to(self.device)
                targets = batch['target'].to(self.device)
                
                predictions = self.binding_predictor(glycan_encoding, protein_encoding, concentration)
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Calculate metrics
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        metrics = calculate_metrics(predictions, targets)
        
        return predictions, targets, metrics
    
    #@property
    #def config(self) -> dict:
        #return self.config


def load_and_predict(checkpoint_path, test_data_path, DEVICE):
    """Load model and make predictions on test data"""
    loader = ModelLoader(checkpoint_path, DEVICE)
    config = loader.config

    
    # Load test data
    fractions_df = pd.read_csv(test_data_path, sep='\t')
    
    # out glycans and proteins datasets are sep by tabs (change yours here)
    glycans_df = pd.read_csv(config.glycans_data_path, sep='\t')
    proteins_df = pd.read_csv(config.proteins_data_path, sep='\t')
    
    # Make predictions
    predictions, targets, metrics = loader.predict(fractions_df, glycans_df, proteins_df)
    
    print(metrics)
    
    return {
        'config': config,
        'metrics': metrics,
        'predictions': predictions.numpy(),
        'targets': targets.numpy()
    }
    
    
def main():
    
    DEVICE = torch.device("cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--testset", type=str, default=None)
    args = parser.parse_args()
    
    #config = TrainingConfig.from_yaml(args.config)
    
    #print('config', config)
    
    #load_and_predict(args.model, './')
    
    print(args.model)
    
    load_and_predict(args.model, args.testset, DEVICE)
    

    
    
if __name__ == "__main__":
    main()