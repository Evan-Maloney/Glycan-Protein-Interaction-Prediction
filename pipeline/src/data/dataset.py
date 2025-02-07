# References:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Docstrings and type-hints generated with GitHub Copilot

import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
from typing import Dict, Tuple
from ..base.encoders import GlycanEncoder, ProteinEncoder

class BindingDataset(Dataset):
    def __init__(self, 
                 data_frame: pd.DataFrame,
                 glycan_encoder: GlycanEncoder,
                 protein_encoder: ProteinEncoder):
        """
        Dataset for glycan-protein binding data
        
        Args:
            data_frame: DataFrame with columns:
                - Glycan SMILE
                - Protein Sequence
                - concentration
                - fraction_bound
            glycan_encoder: Encoder for glycan SMILES
            protein_encoder: Encoder for protein sequences
        """
        self.df = data_frame
        self.glycan_encoder = glycan_encoder
        self.protein_encoder = protein_encoder
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # encode data
        glycan_encoding = self.glycan_encoder.encode_smiles(row['Glycan SMILE'])
        protein_encoding = self.protein_encoder.encode_sequence(row['Protein Sequence'])
        concentration = torch.tensor([row['concentration']], dtype=torch.float32)
        target = torch.tensor([row['fraction_bound']], dtype=torch.float32)
        
        return {
            'glycan_encoding': glycan_encoding,
            'protein_encoding': protein_encoding,
            'concentration': concentration,
            'target': target
        }

def prepare_datasets(
    df: pd.DataFrame,
    val_split: float,
    glycan_encoder: GlycanEncoder,
    protein_encoder: ProteinEncoder
) -> Tuple[BindingDataset, BindingDataset]:
    """
    Prepare train and validation datasets
    
    Args:
        df (pd.DataFrame): Full dataset
        val_split (float): Fraction of data to use for validation
        glycan_encoder (GlycanEncoder): Encoder for glycans
        protein_encoder (ProteinEncoder): Encoder for proteins
    
    Returns:
        Tuple[BindingDataset, BindingDataset]: Train and validation datasets
    """
    full_dataset = BindingDataset(df, glycan_encoder, protein_encoder)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )
    return train_dataset, val_dataset
