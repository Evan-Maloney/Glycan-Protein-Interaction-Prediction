import torch
from typing import List
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from ...base.encoders import ProteinEncoder

class ESM3Encoder(ProteinEncoder):
    def __init__(self):
        super().__init__()
        self.model = ESM3.from_pretrained(ESM3_OPEN_SMALL)
        self.model.eval()
        self._embedding_dim = self.model.embed_dim
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        protein = ESMProtein(sequence=sequence)
        
        with torch.no_grad():
            protein_tensor = self.model.encode(protein)
            output = self.model.forward_and_sample(
                protein_tensor, 
                SamplingConfig(return_per_residue_embeddings=True)
            )
            embedding = output.per_residue_embedding.mean(dim=0)
        
        return embedding
    
    def encode_batch(self, batch_data: List[str]) -> torch.Tensor:
        embeddings = []
        for sequence in batch_data:
            protein = ESMProtein(sequence=sequence)
            with torch.no_grad():
                protein_tensor = self.model.encode(protein)
                output = self.model.forward_and_sample(
                    protein_tensor,
                    SamplingConfig(return_per_residue_embeddings=True)
                )
                embedding = output.per_residue_embedding.mean(dim=0)
                embeddings.append(embedding)
        
        return torch.stack(embeddings)
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def to(self, device):
        self.model = self.model.to(device)
        return self