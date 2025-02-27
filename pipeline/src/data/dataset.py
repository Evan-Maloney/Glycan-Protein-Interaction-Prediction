import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
from typing import Dict, Tuple, List, Any
from tqdm import tqdm

from ..base.encoders import GlycanEncoder

class BindingDataset(Dataset):
    def __init__(self, 
                 data_frame: pd.DataFrame,
                 glycan_encoder: GlycanEncoder = None,
                 precomputed_data: Dict[str, Any] = None):
        """
        Dataset for glycan-protein binding data that caches unique embeddings.

        The dataset precomputes embeddings for each unique glycan and protein sequence
        (instead of re-embedding the same sequence multiple times) to greatly improve performance.
        
        Glycan sequences are encoded using the provided glycan encoder.
        Protein sequences are encoded as one-hot vectors, where each unique protein sequence 
        is assigned a unique one-hot vector.
        
        Args:
            data_frame: DataFrame with required columns, including:
                - 'Glycan SMILE'
                - 'Protein Sequence'
                - 'concentration'
                - 'fraction_bound'
            glycan_encoder: Encoder for glycan SMILES (required if not using precomputed data)
            precomputed_data: Dictionary containing precomputed embeddings for unique sequences.
                              Expected keys: 'glycan_encodings' and 'protein_encodings'.
        """
        self.df = data_frame.copy()
        
        if precomputed_data is not None:
            self.glycan_encodings = precomputed_data['glycan_encodings']
            self.protein_encodings = precomputed_data['protein_encodings']
        else:
            if glycan_encoder is None:
                raise ValueError("Must provide either precomputed data or a glycan encoder")
            # Precompute glycan embeddings
            self.glycan_encodings = self._precompute_unique_embeddings(
                sequences=self.df['Glycan SMILE'].tolist(),
                encoder=glycan_encoder,
                batch_size=8,
                desc="Precomputing unique glycan embeddings"
            )
            # Precompute one-hot protein embeddings
            unique_proteins = sorted(set(self.df['Protein Sequence'].tolist()))
            num_proteins = len(unique_proteins)
            self.protein_encodings = {}
            for i, seq in enumerate(unique_proteins):
                one_hot_vector = torch.zeros(num_proteins, dtype=torch.float)
                one_hot_vector[i] = 1.0
                self.protein_encodings[seq] = one_hot_vector
    
    def _precompute_unique_embeddings(
        self,
        sequences: List[str],
        encoder,
        batch_size: int,
        desc: str = "Precomputing embeddings"
    ) -> Dict[str, torch.Tensor]:
        """
        Precompute embeddings for unique sequences using the provided encoder.
        
        Args:
            sequences: List of sequences (strings) to be encoded.
            encoder: An encoder object that has an `encode_batch` method.
            batch_size: Batch size for processing sequences.
            desc: Description for the progress bar.
            
        Returns:
            A dictionary mapping each unique sequence to its embedding.
        """
        unique_sequences = sorted(set(sequences))
        embeddings = {}
        
        for i in tqdm(range(0, len(unique_sequences), batch_size), desc=desc):
            batch_sequences = unique_sequences[i:i + batch_size]
            with torch.no_grad():
                batch_encodings = encoder.encode_batch(batch_sequences)
            for seq, emb in zip(batch_sequences, batch_encodings):
                embeddings[seq] = emb
        return embeddings
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Return the cached embeddings along with additional data for a given index."""
        row = self.df.iloc[idx]
        glycan_seq = row['Glycan SMILE']
        protein_seq = row['Protein Sequence']
        
        return {
            'glycan_encoding': self.glycan_encodings[glycan_seq],
            'protein_encoding': self.protein_encodings[protein_seq],
            'concentration': torch.tensor([row['concentration']], dtype=torch.float32),
            'target': torch.tensor([row['fraction_bound']], dtype=torch.float32)
        }
    
    def save_precomputed_data(self, path: str):
        """Save the precomputed unique embeddings to disk."""
        torch.save({
            'glycan_encodings': self.glycan_encodings,
            'protein_encodings': self.protein_encodings
        }, path)
    
    @classmethod
    def from_precomputed(cls, data_frame: pd.DataFrame, precomputed_path: str) -> 'BindingDataset':
        """Create a dataset instance using precomputed embeddings."""
        precomputed_data = torch.load(precomputed_path)
        return cls(data_frame, precomputed_data=precomputed_data)

def prepare_datasets(
    df: pd.DataFrame,
    val_split: float,
    glycan_encoder: GlycanEncoder = None,
    precomputed_path: str = None
) -> Tuple[Dataset, Dataset]:
    """
    Prepare train and validation datasets with cached unique embeddings.
    
    Args:
        df: Full dataset DataFrame.
        val_split: Fraction of data to use for validation.
        glycan_encoder: Encoder for glycans (if not using precomputed embeddings).
        precomputed_path: Path to precomputed embeddings (if available).
    
    Returns:
        A tuple containing the training and validation datasets.
    """
    if precomputed_path:
        full_dataset = BindingDataset.from_precomputed(df, precomputed_path)
    else:
        if glycan_encoder is None:
            raise ValueError("Must provide either precomputed_path or a glycan encoder")
        full_dataset = BindingDataset(df, glycan_encoder)
        full_dataset.save_precomputed_data("data/precomputed_data.pt")
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )
    
    return train_dataset, val_dataset
