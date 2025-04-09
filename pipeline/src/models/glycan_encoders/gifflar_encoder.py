# REFERENCES
# https://github.com/BojarLab/glycowork/blob/master/glycowork/ml/inference.py

import torch
from typing import List
from ...base.encoders import GlycanEncoder

class GIFFLAREncoder(GlycanEncoder):
    def __init__(self):
        super().__init__()
        self._embedding_dim = 1024
        self.representation = "IUPAC"
        self.all_embeddings = torch.load(f"src/models/glycan_encoders/assets/GIFFLAR_embeddings.pt")
    
    def encode_smiles(self, iupac: str, device: torch.device) -> torch.Tensor:
        return self.all_embeddings[iupac]
    
    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        embeddings = torch.zeros(len(batch_data), self._embedding_dim)
        for i, iupac in enumerate(batch_data):
            embeddings[i] = self.all_embeddings[iupac]
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def to(self, device):
        return self