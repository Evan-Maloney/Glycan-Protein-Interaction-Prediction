import torch
import numpy as np
from typing import List
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from ...base.encoders import ProteinEncoder

class FeatureBasedProteinEncoder(ProteinEncoder):
    def __init__(self, embedding_dim: int = 12):
        super().__init__()
        self._embedding_dim = embedding_dim
        
    def _extract_features(self, sequence: str) -> np.ndarray:
        if not sequence or not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence):
            return np.zeros(self._embedding_dim)
            
        try:
            analysis = ProteinAnalysis(sequence)
        except Exception:
            return np.zeros(self._embedding_dim)
        
        features = []
        
        # Basic properties
        features.append(len(sequence) / 1000)  # Length
        features.append(analysis.molecular_weight() / 100000)  # Weight
        features.append(analysis.charge_at_pH(7.0))  # Net charge
        features.append(analysis.isoelectric_point() / 14)  # pI
        
        # Amino acid groups
        aa_percent = analysis.get_amino_acids_percent()
        
        # Key amino acid groups for glycan binding
        polar_percent = sum(aa_percent.get(aa, 0.0) for aa in ['N', 'Q', 'S', 'T'])
        basic_percent = sum(aa_percent.get(aa, 0.0) for aa in ['K', 'R', 'H'])
        acidic_percent = sum(aa_percent.get(aa, 0.0) for aa in ['D', 'E'])
        
        features.append(polar_percent)
        features.append(basic_percent)
        features.append(acidic_percent)
        
        # Secondary structure
        helix, turn, sheet = analysis.secondary_structure_fraction()
        features.append(helix)
        features.append(turn)
        features.append(sheet)
        
        # Other key properties
        features.append((analysis.gravy() + 4.5) / 9)  # Hydrophobicity
        
        # Glycosylation sites
        n_glyc_sites = sum(1 for i in range(len(sequence)-2) 
                         if sequence[i] == 'N' and sequence[i+1] != 'P' and sequence[i+2] in ['S', 'T'])
        features.append(min(n_glyc_sites / 10, 1.0))
        
        return np.array(features, dtype=np.float32)
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        features = self._extract_features(sequence)
        return torch.tensor(features, dtype=torch.float32)
    
    def encode_batch(self, batch_data: List[str]) -> torch.Tensor:
        features = np.array([self._extract_features(seq) for seq in batch_data])
        return torch.tensor(features, dtype=torch.float32)
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x