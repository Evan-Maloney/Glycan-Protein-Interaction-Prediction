import torch
import numpy as np
from typing import List, Dict, Any
import xgboost as xgb
from ...base.predictors import BindingPredictor

class XGBBindingPredictor(BindingPredictor):
    """
    XGBoost-based binding predictor that adapts XGBoost to work within the PyTorch framework.
    
    This predictor converts PyTorch tensor inputs to numpy arrays for XGBoost,
    and converts XGBoost's numpy array outputs back to PyTorch tensors.
    """
    def __init__(self, glycan_dim: int, protein_dim: int, params: Dict[str, Any] = None):
        """
        Initialize the XGBoost binding predictor
        
        Args:
            glycan_dim (int): Dimension of glycan embeddings
            protein_dim (int): Dimension of protein embeddings
            params (Dict[str, Any], optional): XGBoost parameters. Defaults to None.
        """
        super().__init__(glycan_dim, protein_dim)
        
        # Default XGBoost parameters
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',  # Use histogram-based algorithm for faster training
            'n_estimators': 100
        }
        
        # Override defaults with provided parameters
        if params is not None:
            self.params.update(params)
            
        # Initialize the XGBoost model
        self.model = xgb.XGBRegressor(**self.params)
        self.is_fitted = False
        
        # Buffer for collecting training data
        self.training_data = {
            'features': [],
            'targets': []
        }
        
    def forward(self, 
               glycan_encoding: torch.Tensor,
               protein_encoding: torch.Tensor,
               concentration: torch.Tensor) -> torch.Tensor:
        """
        Predict binding fraction using XGBoost
        
        Args:
            glycan_encoding (torch.Tensor): Encoded glycan representation
            protein_encoding (torch.Tensor): Encoded protein representation
            concentration (torch.Tensor): Concentration values
            
        Returns:
            torch.Tensor: Predicted fraction bound (values between 0 and 1)
        """
        # Combine features
        features = torch.cat([
            glycan_encoding,
            protein_encoding,
            concentration
        ], dim=-1)
        
        # Convert to numpy for XGBoost
        features_np = features.detach().cpu().numpy()
        
        # If in training mode, collect data for batch training
        if self.training and not self.is_fitted:
            self.training_data['features'].append(features_np)
            return torch.zeros_like(concentration)  # Placeholder during initial training
        
        # Get predictions from XGBoost
        if not self.is_fitted:
            # If model is not fitted yet but we're in eval mode, return zeros
            predictions_np = np.zeros(features_np.shape[0])
        else:
            # Get actual predictions from the model
            predictions_np = self.model.predict(features_np)
        
        # Convert back to torch tensor and ensure proper shape and device
        predictions = torch.tensor(
            predictions_np, 
            dtype=torch.float32, 
            device=concentration.device
        ).view_as(concentration)
        
        # Ensure predictions are in [0, 1] range
        predictions = torch.clamp(predictions, 0, 1)
        
        return predictions
    
    def train_model(self, targets: torch.Tensor) -> None:
        """
        Train the XGBoost model with collected data
        
        Args:
            targets (torch.Tensor): Target values for training
        """
        if not self.training_data['features']:
            return  # No data collected
        
        # Convert collected features to a single numpy array
        X = np.vstack(self.training_data['features'])
        
        # Convert targets to numpy
        y = targets.detach().cpu().numpy()
        
        # Train the XGBoost model
        self.model.fit(X, y)
        
        # Mark as fitted and clear buffer
        self.is_fitted = True
        self.training_data = {
            'features': [],
            'targets': []
        }
        
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances from the trained model
        
        Returns:
            Dict[str, float]: Feature importances
        """
        if not self.is_fitted:
            return {}
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create a dictionary mapping feature indices to importance scores
        feature_importance_dict = {
            f"feature_{i}": float(importance) 
            for i, importance in enumerate(importances)
        }
        
        return feature_importance_dict