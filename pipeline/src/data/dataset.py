import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
from typing import Dict, Tuple, List
from tqdm import tqdm
from ..base.encoders import GlycanEncoder, ProteinEncoder

class BindingDataset(Dataset):
    def __init__(self, 
                 data_frame: pd.DataFrame,
                 glycan_encoder: GlycanEncoder = None,
                 protein_encoder: ProteinEncoder = None,
                 precomputed_data: Dict[str, torch.Tensor] = None):
        """
        Dataset for glycan-protein binding data with precomputed embeddings
        
        Args:
            data_frame: DataFrame with required columns
            glycan_encoder: Encoder for glycan SMILES (only needed if not using precomputed data)
            protein_encoder: Encoder for protein sequences (only needed if not using precomputed data)
            precomputed_data: Dictionary containing precomputed embeddings (if available)
        """
        self.df = data_frame
        
        if precomputed_data is not None:
            self.glycan_encodings = precomputed_data['glycan_encodings']
            self.protein_encodings = precomputed_data['protein_encodings']
            self.glycan_to_index = precomputed_data['glycan_to_index']
            self.protein_to_index = precomputed_data['protein_to_index']
        else:
            if glycan_encoder is None or protein_encoder is None:
                raise ValueError("Must provide either precomputed data or both encoders")
            
            # Compute the mappings and embeddings
            self.glycan_to_index, self.glycan_encodings = self._precompute_unique_glycans(glycan_encoder)
            self.protein_to_index, self.protein_encodings = self._precompute_unique_proteins(protein_encoder)
    
    def _precompute_unique_glycans(self, glycan_encoder: GlycanEncoder) -> Tuple[Dict[str, int], torch.Tensor]:
        """
        Precompute embeddings for unique glycans in the dataset
        
        Args:
            glycan_encoder: Encoder for glycan SMILES
            
        Returns:
            Tuple of (mapping from SMILE to index, tensor of glycan embeddings)
        """
        print("Precomputing unique glycan embeddings...")
        
        # Get unique glycan SMILES
        unique_glycans = self.df['Glycan SMILE'].unique()
        glycan_to_index = {smile: i for i, smile in enumerate(unique_glycans)}
        
        # Compute embeddings for each unique glycan
        glycan_encodings = []
        for smile in tqdm(unique_glycans):
            with torch.no_grad():
                encoding = glycan_encoder.encode_smiles(smile)
                glycan_encodings.append(encoding)
        
        return glycan_to_index, torch.stack(glycan_encodings)
    
    def _precompute_unique_proteins(self, protein_encoder: ProteinEncoder) -> Tuple[Dict[str, int], torch.Tensor]:
        """
        Precompute embeddings for unique proteins in the dataset
        
        Args:
            protein_encoder: Encoder for protein sequences
            
        Returns:
            Tuple of (mapping from sequence to index, tensor of protein embeddings)
        """
        print("Precomputing unique protein embeddings...")
        
        # Get unique protein sequences
        unique_proteins = self.df['Protein Sequence'].unique()
        protein_to_index = {seq: i for i, seq in enumerate(unique_proteins)}
        
        # Compute embeddings for unique proteins in batches
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
        """Get precomputed embeddings and other data for an index"""
        row = self.df.iloc[idx]
        
        # Get the corresponding embedding using the mapping
        glycan_idx = self.glycan_to_index[row['Glycan SMILE']]
        protein_idx = self.protein_to_index[row['Protein Sequence']]
        
        return {
            'glycan_encoding': self.glycan_encodings[glycan_idx],
            'protein_encoding': self.protein_encodings[protein_idx],
            'concentration': torch.tensor([row['concentration']], dtype=torch.float32),
            'target': torch.tensor([row['fraction_bound']], dtype=torch.float32)
        }
    
    def save_precomputed_data(self, path: str):
        """Save precomputed embeddings and mappings to disk"""
        torch.save({
            'glycan_encodings': self.glycan_encodings,
            'protein_encodings': self.protein_encodings,
            'glycan_to_index': self.glycan_to_index,
            'protein_to_index': self.protein_to_index
        }, path)
    
    @classmethod
    def from_precomputed(cls, data_frame: pd.DataFrame, precomputed_path: str) -> 'BindingDataset':
        """Create dataset from precomputed embeddings"""
        precomputed_data = torch.load(precomputed_path)
        return cls(data_frame, precomputed_data=precomputed_data)

def prepare_datasets(
    df: pd.DataFrame,
    val_split: float,
    glycan_encoder: GlycanEncoder = None,
    protein_encoder: ProteinEncoder = None,
    precomputed_path: str = None
) -> Tuple[Dataset, Dataset]:
    """
    Prepare train and validation datasets
    
    Args:
        df: Full dataset DataFrame
        val_split: Fraction of data to use for validation
        glycan_encoder: Encoder for glycans (if not using precomputed)
        protein_encoder: Encoder for proteins (if not using precomputed)
        precomputed_path: Path to precomputed embeddings (if available)
    
    Returns:
        Tuple of train and validation datasets
    """
    if precomputed_path:
        full_dataset = BindingDataset.from_precomputed(df, precomputed_path)
    else:
        if glycan_encoder is None or protein_encoder is None:
            raise ValueError("Must provide either precomputed_path or both encoders")
        full_dataset = BindingDataset(df, glycan_encoder, protein_encoder)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )
    
    return train_dataset, val_dataset