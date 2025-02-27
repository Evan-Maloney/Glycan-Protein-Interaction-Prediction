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
        # Create all protein objects at once
        proteins = [ESMProtein(sequence=seq) for seq in batch_data]
        
        with torch.no_grad():
            # Get tokens for all sequences
            tokens = self.model._tokenize([p.sequence for p in proteins])
            
            # Create the sequence_id mask for padding
            sequence_id = (tokens != self.model.tokenizer.pad_token_id)
            
            # Forward pass through the model with the batch
            output = self.model.forward(
                sequence_tokens=tokens,
                sequence_id=sequence_id
            )
            
            # Mean pool over sequence length for each protein
            # output.embeddings shape: [batch_size, seq_length, embedding_dim]
            embeddings = output.embeddings.mean(dim=1)  # -> [batch_size, embedding_dim]
            
            return embeddings
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def to(self, device):
        self.model = self.model.to(device)
        return self