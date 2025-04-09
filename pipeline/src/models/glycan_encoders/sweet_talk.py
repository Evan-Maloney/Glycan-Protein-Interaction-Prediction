import torch
from torch import nn
from typing import List
import numpy as np
import pandas as pd
import re
from ...base.encoders import GlycanEncoder

class SweetTalkGlycanEncoder(GlycanEncoder):
    def __init__(self, model_path: str = './data/sweet_talk_model_weights/glycanML_SweetTalk_large2.pt', embedding_dim: int = 70, device: str = 'cpu'):
        super().__init__()
        self._embedding_dim = embedding_dim
        self.device = torch.device(device)
        
        # Load the pretrained SweetTalk model
        self.model = torch.load(model_path, weights_only=False, map_location=self.device)
        self.model.eval()
        
        # Load glycoletter library
        self.lib_all_long = self._load_glycoletter_library()
        self.num_classes = embedding_dim #len(self.lib_all_long)+1
        
        # add out own linear layer on top to adjust to our task
        #self.projection = nn.Linear(self.num_classes, self.num_classes)
        self.projection = nn.Sequential(
            nn.Linear(self.num_classes, self.num_classes*2),
            nn.ReLU(),
            nn.Linear(self.num_classes*2, self.num_classes)
        )
        ## Optional projection layer if embedding dimension differs from model
        #if embedding_dim != 128:
            #self.projection = nn.Linear(128, embedding_dim)
        #else:
            #self.projection = nn.Identity()
            
    def _convert_iupacs(self, iupac_batch):
        converted = []
        
        for glycan in iupac_batch:
       
            glycan = glycan.replace('α', 'a').replace('β', 'b')
            
            # remove number before any os and op
            glycan = re.sub(r'(\D+)\d+OS', r'\1OS', glycan)
            glycan = re.sub(r'(\D+)\d+OP', r'\1OP', glycan)
            
            # remove final anomeric state and spacer     str.rsplit('(', 1)[0]
            glycan = glycan.rsplit('(', 1)[0]
            
            converted.append(glycan)
        
        return converted
            
    def _motif_find(self, s):
        """converts a IUPACcondensed-ish glycan into a list of overlapping, asterisk-separated glycowords"""
        b = s.split('(')
        b = [k.split(')') for k in b]
        b = [item for sublist in b for item in sublist]
        b = [k.strip('[') for k in b]
        b = [k.strip(']') for k in b]
        b = [k.replace('[', '') for k in b]
        b = [k.replace(']', '') for k in b]
        b = ['*'.join(b[i:i+5]) for i in range(0, len(b)-4, 2)]
        return b
            
    def _process_glycans(self, glycan_list):
        """converts list of glycans into a list of lists of glycowords"""
        glycan_motifs = [self._motif_find(k) for k in glycan_list]
        glycan_motifs = [[i.split('*') for i in k] for k in glycan_motifs]
        return glycan_motifs
        
    def _load_glycoletter_library(self):
        """Loads glycoletter vocabulary from dataset."""
        df_all_long = pd.read_csv('./data/Glycan-Structures-CFG611.txt', sep='\t').IUPAC.values.tolist()
        df_all_long = self._process_glycans(df_all_long)
        df_all_long = [item for sublist in df_all_long for item in sublist]
        return sorted(set([item for sublist in df_all_long for item in sublist]))
    
    def _motif_find(self, glycan_str: str) -> List[str]:
        """Extracts overlapping glycowords from an IUPAC glycan representation."""
        parts = glycan_str.replace('[', '').replace(']', '').split('(')
        parts = [p.split(')') for p in parts]
        parts = [item for sublist in parts for item in sublist]
        return ['*'.join(parts[i:i+5]) for i in range(len(parts)-4)]
    
    def _tokenize_glycan(self, glycan_str: str) -> List[int]:
        """Converts glycan string into a list of token indices."""
        glycowords = self._motif_find(glycan_str)
        token_indices = [self.lib_all_long.index(gw) for gw in glycowords if gw in self.lib_all_long]
        return token_indices if token_indices else [0]  # Default token for unknowns
    
    def encode_iupac(self, iupac_str: str, device: torch.device) -> torch.Tensor:
        """Encodes a single IUPAC glycan string into an embedding vector."""
        
        token_indices = self._tokenize_glycan(iupac_str)
        input_tensor = torch.LongTensor(token_indices).unsqueeze(1).to(device)  # Shape: [seq_len, batch_size=1]
        
        
        #print(f"input_seq shape: {input_tensor.shape}")  # Debugging shape
    
        embedded = self.model.encoder(input_tensor)  # (seq_len, batch_size, hidden_size)
        #print(f"embedded shape before BN: {embedded.shape}")
        
        # Reshape for BatchNorm1d: Remove seq_len=1 if it's present
        #embedded = embedded.squeeze(0)  # Shape: (batch_size=1, hidden_size=128)

        # Permute to (batch_size, hidden_size, seq_len) for BatchNorm1d
        #embedded = embedded.permute(1, 2, 0)
        #print(f"embedded shape after permute: {embedded.shape}")
        
        #print('embedded shape right before bn1 1', embedded.shape)
        #embedded = embedded.squeeze(0)
        #print('embedded shape right before bn1 2', embedded.shape)
        #embedded = self.model.bn1(embedded)  
        #print(f"embedded shape after BN: {embedded.shape}")

        # Restore to (seq_len, batch_size, hidden_size)
        # Reshape back if needed
        #embedded = embedded.unsqueeze(0)  # Restore (seq_len=1, batch_size=1, hidden_size=128)

        outputs, (h_n, c_n) = self.model.gru(embedded, None)
        
        #print('outputs shape', outputs.shape)
        
        logits = self.model.logits_fc(outputs)
        #logits = logits.transpose(0, 1).contiguous()
        #print('logits shape', logits.shape)
        logits_flatten = logits.view(-1, self.num_classes)
        
        
        #----------
        
        #with torch.no_grad():
         #   embedded = self.model.encoder(input_tensor)  # Get encoded representations
         #   embedded = self.model.bn1(embedded)  # Apply batch norm
         #   _, (h_n, _) = self.model.gru(embedded)  # Extract hidden states
            
            # Handle bidirectional GRU by concatenating last forward and backward states
          #  final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        #print('iupac encoding shape', self.projection(logits_flatten.squeeze(0)).shape)
        
        return self.projection(logits_flatten.squeeze(0))  # Apply projection if necessary
    
    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        """Encodes a batch of IUPAC glycan strings into embeddings."""
        
        # convert the iupacs to sweet talk conversion
        #batch_data = self._convert_iupacs(batch_data)
        
        iupac_encoding_batch = torch.stack([self.encode_iupac(iupac, device) for iupac in batch_data])
        
        #print('iupac batch before:', iupac_encoding_batch.shape)
        #iupac_encoding_batch = self.model.bn1(iupac_encoding_batch)
        #print('iupac batch after:', iupac_encoding_batch.shape)
        return iupac_encoding_batch
    
    def encode_smiles(self, smiles: str) -> torch.Tensor:
        """Not implemented since this model does not use SMILES encoding."""
        raise NotImplementedError("SMILES encoding is not supported in SweetTalkGlycanEncoder.")
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
