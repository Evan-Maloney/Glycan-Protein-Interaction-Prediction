# References:
# https://stackoverflow.com/questions/72077539/should-i-inherit-from-both-nn-module-and-abc
# Abstract base class approach inspired by FlairNLP: https://github.com/flairNLP/flair/blob/master/flair/embeddings/base.py
# Docstrings and type-hints generated with GitHub Copilot
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# https://docs.python.org/3/library/abc.html

from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import List

class GlycanEncoder(nn.Module, ABC):
    @abstractmethod
    def encode_smiles(self, smiles: str) -> torch.Tensor:
        """
        Encode a SMILES string to a fixed-length tensor
        
        Args:
            smiles (str): SMILES string representing a glycan molecule
            
        Returns:
            torch.Tensor: Encoded representation of the glycan
        """
        pass
    
    @abstractmethod
    def encode_batch(self, batch_data: List[str]) -> torch.Tensor:
        """
        Encode a batch of SMILES strings
        
        Args:
            batch_data (List[str]): List of SMILES strings
            
        Returns:
            torch.Tensor: Batch of encoded representations
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of the encoding"""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder"""
        raise NotImplementedError("Subclasses must implement forward()")


class ProteinEncoder(nn.Module, ABC):
    @abstractmethod
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Encode an amino acid sequence to a fixed-length tensor
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            torch.Tensor: Encoded representation of the protein
        """
        pass
    
    @abstractmethod
    def encode_batch(self, batch_data: List[str]) -> torch.Tensor:
        """
        Encode a batch of amino acid sequences
        
        Args:
            batch_data (List[str]): List of amino acid sequences
            
        Returns:
            torch.Tensor: Batch of encoded representations
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of the encoding"""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder"""
        raise NotImplementedError("Subclasses must implement forward()")