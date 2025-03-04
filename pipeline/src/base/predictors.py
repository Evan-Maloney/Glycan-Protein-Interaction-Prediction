# References:
# https://stackoverflow.com/questions/72077539/should-i-inherit-from-both-nn-module-and-abc
# Abstract base class approach inspired by FlairNLP: https://github.com/flairNLP/flair/blob/master/flair/embeddings/base.py
# Docstrings and type-hints generated with GitHub Copilot
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# https://docs.python.org/3/library/abc.html

from abc import ABC, abstractmethod
import torch
from torch import nn

class BindingPredictor(nn.Module, ABC):
    def __init__(self, glycan_dim: int, protein_dim: int):
        """
        Initialize the binding predictor
        
        Args:
            glycan_dim (int): Dimension of glycan embeddings
            protein_dim (int): Dimension of protein embeddings
        """
        super().__init__()
        self.glycan_dim = glycan_dim
        self.protein_dim = protein_dim
    
    @abstractmethod
    def forward(self, 
                glycan_encoding: torch.Tensor,
                protein_encoding: torch.Tensor,
                concentration: torch.Tensor) -> torch.Tensor:
        """
        Predict binding fraction
        
        Args:
            glycan_encoding (torch.Tensor): Encoded glycan representation
            protein_encoding (torch.Tensor): Encoded protein representation
            concentration (torch.Tensor): Concentration values
            
        Returns:
            torch.Tensor: Predicted fraction bound (values between 0 and 1)
        """
        pass