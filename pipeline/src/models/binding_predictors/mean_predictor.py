# This dummy predictor is for testing purposes only

import torch
from ...base.predictors import BindingPredictor

class MeanPredictor(BindingPredictor):
    def __init__(self, glycan_dim: int, protein_dim: int):
        super().__init__(glycan_dim, protein_dim)
        self.linear = torch.nn.Linear(glycan_dim + protein_dim + 1, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, 
                glycan_encoding: torch.Tensor,
                protein_encoding: torch.Tensor,
                concentration: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([
            glycan_encoding,
            protein_encoding,
            concentration
        ], dim=-1)
        
        output_shape = self.linear(concat).shape  # Determine the shape
        constant_value = 0.016757234366309902
        mean_predict = torch.full(output_shape, constant_value, requires_grad=True)
        #print('mean', mean_predict)

        # Create a tensor with the required shape and constant value
        return mean_predict