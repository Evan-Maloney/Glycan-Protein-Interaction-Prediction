# REFERNCES
# Ankh repo: https://github.com/agemagician/Ankh/tree/main
# GIFFLAR implementation of Ankh: https://github.com/BojarLab/GIFFLAR/blob/main/experiments/protein_encoding.py

import torch
from typing import List
from ...base.encoders import ProteinEncoder
from transformers import T5EncoderModel, AutoTokenizer

class AnkhEncoder(ProteinEncoder):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")
        self.model = T5EncoderModel.from_pretrained("ElnaggarLab/ankh-base")
        self.model.eval()
        self._embedding_dim = self.model.config.d_model
    
    def encode_sequence(self, sequence: str, device: torch.device) -> torch.Tensor:
        # Get tokens for sequence
        protein = list(sequence)
        
        outputs = self.tokenizer.encode_plus(
            protein,
            add_special_tokens=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        
        input_ids = outputs["input_ids"].to(device)
        attention_mask = outputs["attention_mask"].to(device)
        with torch.no_grad():
            embeddings = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        # Mean pool over sequence length for each protein
        # embeddings.last_hidden_state shape: [1, seq_length, embedding_dim]
        return embeddings.last_hidden_state.squeeze(0).mean(dim=0)
    
    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        protein_sequences = [list(sequence) for sequence in batch_data]
        
        # Get tokens for all sequences
        outputs = self.tokenizer.batch_encode_plus(protein_sequences, 
                                add_special_tokens=True, 
                                padding=True, 
                                is_split_into_words=True, 
                                return_tensors="pt")
        
        input_ids = outputs["input_ids"].to(device)
        attention_mask = outputs["attention_mask"].to(device)
        with torch.no_grad():
            embeddings = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
        # Mean pool over sequence length for each protein
        # embeddings.last_hidden_state shape: [batch_size, seq_length, embedding_dim]
        return embeddings.last_hidden_state.mean(dim=1) # -> [batch_size, embedding_dim]
        
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def to(self, device):
        self.model = self.model.to(device)
        return self