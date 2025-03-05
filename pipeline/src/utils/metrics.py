# References:
# Function type-hints and docstrings generated with GitHub Copilot
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

import torch
from typing import Dict
import numpy as np
from scipy.stats import pearsonr

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