import torch
from typing import List
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from ...base.encoders import ProteinEncoder

class ESMCEncoder(ProteinEncoder):
    def __init__(self):
        super().__init__()
        self.model = ESMC.from_pretrained("esmc_300m")
        self.model.eval()
        self._embedding_dim = self.model.embed.embedding_dim
    
    def encode_sequence(self, sequence: str, device: torch.device) -> torch.Tensor:
        protein = ESMProtein(sequence=sequence)
        
        with torch.no_grad():
            protein_tensor = self.model.encode(protein)
            output = self.model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            # Handle the batch dimension properly
            # output.embeddings shape is [batch_size=1, seq_length, embedding_dim]
            embedding = output.embeddings.squeeze(0).mean(dim=0)
        
        return embedding
    
    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        # Create all protein objects at once
        proteins = [ESMProtein(sequence=seq) for seq in batch_data]
        
        with torch.no_grad():
            # Get tokens for all sequences
            tokens = self.model._tokenize([p.sequence for p in proteins])
            
            # Create the sequence_id mask for padding
            sequence_id = (tokens != self.model.tokenizer.pad_token_id)

            tokens = tokens.to(device)
            sequence_id = sequence_id.to(device)

            
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