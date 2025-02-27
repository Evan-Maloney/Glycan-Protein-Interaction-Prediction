import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
from typing import Dict, Tuple, List
from tqdm import tqdm
from ..base.encoders import GlycanEncoder, ProteinEncoder

class BindingDataset(Dataset):
    def __init__(self, 
                 data_frame: pd.DataFrame,
                 glycan_encoder: GlycanEncoder,
                 protein_encoder: ProteinEncoder):
        """
        Dataset for glycan-protein binding data with encodings for unique molecules
        
        Args:
            data_frame: DataFrame with required columns
            glycan_encoder: Encoder for glycan SMILES
            protein_encoder: Encoder for protein sequences
        """
        self.df = data_frame
        
        # Encode unique glycans and proteins
        self.glycan_to_index, self.glycan_encodings = self._encode_unique_glycans(glycan_encoder)
        self.protein_to_index, self.protein_encodings = self._encode_unique_proteins(protein_encoder)
    
    def _encode_unique_glycans(self, glycan_encoder: GlycanEncoder) -> Tuple[Dict[str, int], torch.Tensor]:
        """
        Encode each unique glycan in the dataset once
        
        Args:
            glycan_encoder: Encoder for glycan SMILES
            
        Returns:
            Tuple of (mapping from SMILE to index, tensor of glycan encodings)
        """
        print("Encoding unique glycans...")
        
        # Get unique glycan SMILES
        unique_glycans = self.df['Glycan SMILE'].unique()
        glycan_to_index = {smile: i for i, smile in enumerate(unique_glycans)}
        
        # Encode each unique glycan
        glycan_encodings = []
        for smile in tqdm(unique_glycans):
            with torch.no_grad():
                encoding = glycan_encoder.encode_smiles(smile)
                glycan_encodings.append(encoding)
        
        return glycan_to_index, torch.stack(glycan_encodings)
    
    def _encode_unique_proteins(self, protein_encoder: ProteinEncoder) -> Tuple[Dict[str, int], torch.Tensor]:
        """
        Encode each unique protein in the dataset once
        
        Args:
            protein_encoder: Encoder for protein sequences
            
        Returns:
            Tuple of (mapping from sequence to index, tensor of protein encodings)
        """
        print("Encoding unique proteins...")
        
        # Get unique protein sequences
        unique_proteins = self.df['Protein Sequence'].unique()
        protein_to_index = {seq: i for i, seq in enumerate(unique_proteins)}
        
        # Encode unique proteins in batches
        protein_encodings = []
        batch_size = 2  # Adjust based on your memory constraints
        
        for i in tqdm(range(0, len(unique_proteins), batch_size)):
            batch_sequences = unique_proteins[i:i + batch_size]
            with torch.no_grad():
                batch_encodings = protein_encoder.encode_batch(batch_sequences)
                if isinstance(batch_encodings, list):
                    protein_encodings.extend(batch_encodings)
                else:
                    # If encode_batch returns a tensor
                    for j in range(len(batch_sequences)):
                        protein_encodings.append(batch_encodings[j])
        
        return protein_to_index, torch.stack(protein_encodings)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get encodings and other data for an index"""
        row = self.df.iloc[idx]
        
        # Get the corresponding encoding using the mapping
        glycan_idx = self.glycan_to_index[row['Glycan SMILE']]
        protein_idx = self.protein_to_index[row['Protein Sequence']]
        
        return {
            'glycan_encoding': self.glycan_encodings[glycan_idx],
            'protein_encoding': self.protein_encodings[protein_idx],
            'concentration': torch.tensor([row['concentration']], dtype=torch.float32),
            'target': torch.tensor([row['fraction_bound']], dtype=torch.float32)
        }

def prepare_datasets(
    df: pd.DataFrame,
    val_split: float,
    glycan_encoder: GlycanEncoder,
    protein_encoder: ProteinEncoder
) -> Tuple[Dataset, Dataset]:
    """
    Prepare train and validation datasets
    
    Args:
        df: Full dataset DataFrame
        val_split: Fraction of data to use for validation
        glycan_encoder: Encoder for glycans
        protein_encoder: Encoder for proteins
    
    Returns:
        Tuple of train and validation datasets
    """
    full_dataset = BindingDataset(df, glycan_encoder, protein_encoder)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )
    
    return train_dataset, val_dataset