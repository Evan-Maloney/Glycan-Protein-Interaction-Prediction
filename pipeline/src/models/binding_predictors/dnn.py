# References:
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html
# https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
# Docstrings and type-hints generated with GitHub Copilot

import torch
from torch import nn
from typing import List
from ...base.predictors import BindingPredictor

class DNNBindingPredictor(BindingPredictor):
    def __init__(self, glycan_dim: int, protein_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        """
        Simple DNN for binding prediction
        
        Args:
            glycan_dim (int): Dimension of glycan embeddings
            protein_dim (int): Dimension of protein embeddings
            hidden_dims (List[int]): List of hidden layer dimensions
        """
        super().__init__(glycan_dim, protein_dim)
                
        input_dim = glycan_dim + protein_dim + 1 # total input feature size
        
        self.network = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(input_dim if i==0 else hidden_dims[i-1], hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ) for i, hidden_dim in enumerate(hidden_dims)],
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                glycan_encoding: torch.Tensor,
                protein_encoding: torch.Tensor,
                concentration: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            glycan_encoding (torch.Tensor): Encoded glycan representation
            protein_encoding (torch.Tensor): Encoded protein representation
            concentration (torch.Tensor): Concentration values
            
        Returns:
            torch.Tensor: Predicted fraction bound (values between 0 and 1)
        """
        # combine input features
        x = torch.cat([
            glycan_encoding,
            protein_encoding,
            concentration
        ], dim=-1)
        
        return self.network(x)