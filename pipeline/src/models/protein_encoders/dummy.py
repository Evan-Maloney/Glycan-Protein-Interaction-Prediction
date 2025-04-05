import torch
from typing import List
from ...base.encoders import ProteinEncoder

class DummyProteinEncoder(ProteinEncoder):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self._embedding_dim = embedding_dim
    
    def encode_sequence(self, sequence: str, device: torch.device) -> torch.Tensor:
        """Returns random embedding for any protein sequence"""
        return torch.randn(self._embedding_dim).to(device)
    
    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        """Returns batch of random embeddings"""
        return torch.randn(len(batch_data), self._embedding_dim).to(device)
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For compatibility with nn.Module
        return x