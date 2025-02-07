# References:
# https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
# https://github.com/seyonechithrananda/bert-loves-chemistry/blob/master/README.md
# https://github.com/seyonechithrananda/bert-loves-chemistry/blob/695bc28cbaa0b00410711f1b2ab5953cd668530d/chemberta/visualization/viz_utils.py#L109

import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
from ...base.encoders import GlycanEncoder

class ChemBERTaEncoder(GlycanEncoder):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.model.eval()
        self._embedding_dim = self.model.config.hidden_size
    
    def encode_smiles(self, smiles: str) -> torch.Tensor:
        inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # ChemBERTa outputs one embedding per token, so we take the mean
            # (should probably find research to back this up as a valid strategy)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        
        return embedding
    
    def encode_batch(self, batch_data: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # ChemBERTa outputs one embedding per token, so we take the mean
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def to(self, device):
        self.model = self.model.to(device)
        return self