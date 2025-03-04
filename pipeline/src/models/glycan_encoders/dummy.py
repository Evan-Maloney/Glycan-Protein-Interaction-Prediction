# # This dummy encoder is for testing purposes only

import torch
from typing import List
from ...base.encoders import GlycanEncoder

class DummyGlycanEncoder(GlycanEncoder):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self._embedding_dim = embedding_dim
    
    def encode_smiles(self, smiles: str) -> torch.Tensor:
        return torch.randn(self._embedding_dim)
    
    def encode_batch(self, batch_data: List[str]) -> torch.Tensor:
        return torch.randn(len(batch_data), self._embedding_dim)
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x