# REFERNCES
# ProstT5 repo: https://github.com/mheinzinger/ProstT5
# GIFFLAR implementation of ProstT5: https://github.com/BojarLab/GIFFLAR/blob/main/experiments/protein_encoding.py

import torch
from typing import List
from ...base.encoders import ProteinEncoder
from transformers import T5EncoderModel, T5Tokenizer
import re

class ProstT5Encoder(ProteinEncoder):
    def __init__(self):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.model.eval()
        self._embedding_dim = self.model.config.d_model
    
    def encode_sequence(self, sequence: str, device: torch.device) -> torch.Tensor:
        # replace all rare/ambiguous amino acids by X (3Di sequences do not have those) and introduce white-space between all sequences (AAs and 3Di)
        protein = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        protein = "<AA2fold>" + " " + protein.upper()
        
        # tokenize sequence
        ids = self.tokenizer.encode_plus(
            protein,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        input_ids = ids["input_ids"].to(device),
        attention_mask = ids["attention_mask"].to(device)
        with torch.no_grad():
            embeddings = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        # Mean pool over sequence length for each protein
        # embeddings.last_hidden_state shape: [1, seq_length, embedding_dim]
        return embeddings.last_hidden_state.squeeze(0).mean(dim=0) 
    
    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        # replace all rare/ambiguous amino acids by X (3Di sequences do not have those) and introduce white-space between all sequences (AAs and 3Di)
        proteins = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in batch_data]
        proteins = ["<AA2fold>" + " " + s.upper() for s in proteins]
        
        # tokenize sequences and pad up to the longest sequence in the batch
        ids = self.tokenizer.batch_encode_plus(
            proteins,
            add_special_tokens=True,
            padding="longest",
            return_tensors='pt'
        )
        
        input_ids = ids["input_ids"].to(device)
        attention_mask = ids["attention_mask"].to(device)
        with torch.no_grad():
            embeddings = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
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