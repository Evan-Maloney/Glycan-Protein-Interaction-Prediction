# NOTE: calculating embeddings using embeddings using GIFFLAR takes very long (> 30 mins for our 600 glycans)
# So, I calculated the embeddings using the code availbe at this repo: https://github.com/loganwoudstra/GIFFLAR
# and cached them locally

import torch
from typing import List
from ...base.encoders import GlycanEncoder

class GIFFLAREncoder(GlycanEncoder):
    def __init__(self):
        super().__init__()
        self._embedding_dim = 1024
        self.all_embeddings = torch.load(f"src/models/glycan_encoders/assets/GIFFLAR_embeddings.pt")
        
    def encode_iupac(self, iupac: str) -> torch.Tensor:
        """Not implemented since this model does not use IUPAC encoding."""
        raise NotImplementedError("IUPAC encoding is not supported in SweetNetGlycanEncoder.")
    
    def encode_smiles(self, smiles: str, device: torch.device) -> torch.Tensor:
        return self.all_embeddings[smiles]
    
    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        embeddings = torch.zeros(len(batch_data), self._embedding_dim)
        for i, smiles in enumerate(batch_data):
            embeddings[i] = self.all_embeddings[smiles]
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def to(self, device):
        return self