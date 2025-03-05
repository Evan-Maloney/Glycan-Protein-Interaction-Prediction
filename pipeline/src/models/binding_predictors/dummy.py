# This dummy predictor is for testing purposes only

import torch
from ...base.predictors import BindingPredictor

class DummyBindingPredictor(BindingPredictor):
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

        return self.sigmoid(self.linear(concat))