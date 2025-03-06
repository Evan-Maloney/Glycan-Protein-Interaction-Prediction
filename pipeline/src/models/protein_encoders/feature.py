import torch
import numpy as np
from typing import List, Dict
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from ...base.encoders import ProteinEncoder

class BiophysicalProteinEncoder(ProteinEncoder):
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self._embedding_dim = embedding_dim
        
        self.polar_aas = ['N', 'Q', 'S', 'T']  
        self.basic_aas = ['K', 'R']  
        self.acidic_aas = ['D', 'E']  
        self.aromatic_aas = ['F', 'Y', 'W']  
        self.aliphatic_aas = ['A', 'V', 'L', 'I', 'M'] 
        self.small_aas = ['G', 'A', 'S'] 
        
        # Other properties relevant for encoding
        self.all_aas = list("ACDEFGHIKLMNPQRSTVWY")
        
        # Define pH values for charge calculations
        self.ph_values = [5.0, 6.0, 7.0, 8.0]
        
    def compute_protein_features(self, seq: str) -> Dict:
        """
        Compute various biophysical and biochemical features of a protein sequence.
        
        Args:
            seq: A protein sequence as a string of amino acid one-letter codes
        
        Returns:
            Dictionary of features
        """
        # Clean sequence - replace rare amino acids with X
        seq = ''.join(['X' if aa not in self.all_aas else aa for aa in seq])
        
        # Skip empty sequences or those with too many unknown AAs
        if len(seq) == 0 or seq.count('X') / len(seq) > 0.2:
            return self._get_default_features()
        
        try:
            # Use Biopython's ProteinAnalysis for calculations
            analysis = ProteinAnalysis(seq.replace('X', ''))
            features = {}
            
            # Sequence length and molecular properties
            features['length'] = len(seq)
            features['mw'] = analysis.molecular_weight() / 10000  # Normalize
            features['instability_index'] = min(analysis.instability_index() / 100, 1.0)  # Normalize
            
            # Charge at different pH values
            for ph in self.ph_values:
                features[f'charge_pH{ph}'] = analysis.charge_at_pH(ph) / 10  # Normalize
            
            # Amino acid composition
            aa_percent = analysis.get_amino_acids_percent()
            
            # Individual amino acid percentages (important ones)
            for aa_group, group_name in [
                (self.polar_aas, 'polar'),
                (self.basic_aas, 'basic'),
                (self.acidic_aas, 'acidic'),
                (self.aromatic_aas, 'aromatic'),
                (self.aliphatic_aas, 'aliphatic'),
                (self.small_aas, 'small')
            ]:
                features[f'frac_{group_name}'] = sum(aa_percent.get(aa, 0.0) for aa in aa_group)
            
            # Calculate specific scores for glycan binding potential
            features['aromatic_binding_score'] = sum(aa_percent.get(aa, 0.0) for aa in self.aromatic_aas)
            features['h_bond_potential'] = sum(aa_percent.get(aa, 0.0) for aa in self.polar_aas + self.basic_aas)
            
            # Other physical properties
            features['aromaticity'] = analysis.aromaticity()
            features['hydrophobicity'] = analysis.gravy()  # GRAVY score
            
            # Secondary structure prediction (approximate)
            secondary_struct = analysis.secondary_structure_fraction()
            features['helix_fraction'] = secondary_struct[0]
            features['turn_fraction'] = secondary_struct[1]
            features['sheet_fraction'] = secondary_struct[2]
            
            return features
            
        except Exception as e:
            # Return default features if analysis fails
            print(f"Error analyzing sequence: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict:
        """Return default feature values when analysis fails"""
        features = {
            'length': 0,
            'mw': 0,
            'instability_index': 0
        }
        
        # Add default values for all features
        for ph in self.ph_values:
            features[f'charge_pH{ph}'] = 0
            
        for group_name in ['polar', 'basic', 'acidic', 'aromatic', 'aliphatic', 'small']:
            features[f'frac_{group_name}'] = 0
            
        features['aromatic_binding_score'] = 0
        features['h_bond_potential'] = 0
        features['aromaticity'] = 0
        features['hydrophobicity'] = 0
        features['helix_fraction'] = 0
        features['turn_fraction'] = 0
        features['sheet_fraction'] = 0
        
        return features
    
    def _features_to_tensor(self, features: Dict) -> torch.Tensor:
        """Convert feature dictionary to fixed-dimension tensor"""
        # Define a fixed order of features to ensure consistent tensor dimensions
        feature_order = [
            'length', 'mw', 'instability_index', 
            'charge_pH5.0', 'charge_pH6.0', 'charge_pH7.0', 'charge_pH8.0',
            'frac_polar', 'frac_basic', 'frac_acidic', 'frac_aromatic', 'frac_aliphatic', 'frac_small',
            'aromatic_binding_score', 'h_bond_potential', 'aromaticity', 'hydrophobicity',
            'helix_fraction', 'turn_fraction', 'sheet_fraction'
        ]
        
        # Extract values in order (default to 0 if missing)
        values = [features.get(f, 0.0) for f in feature_order]
        
        # Convert to tensor and normalize
        tensor = torch.tensor(values, dtype=torch.float32)
        
        # If embedding_dim is larger than our feature set, pad with zeros
        if self._embedding_dim > len(values):
            padding = torch.zeros(self._embedding_dim - len(values), dtype=torch.float32)
            tensor = torch.cat([tensor, padding])
        
        # If embedding_dim is smaller, we need to project down the features
        elif self._embedding_dim < len(values):
            # Simple downsampling by averaging groups of features
            group_size = len(values) // self._embedding_dim
            remainder = len(values) % self._embedding_dim
            
            result = []
            idx = 0
            
            for i in range(self._embedding_dim):
                # Determine group size for this dimension (distribute remainder)
                current_group_size = group_size + (1 if i < remainder else 0)
                # Take the mean of the group
                group_mean = tensor[idx:idx+current_group_size].mean()
                result.append(group_mean)
                idx += current_group_size
                
            tensor = torch.tensor(result, dtype=torch.float32)
            
        return tensor
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Encode a single protein sequence into a tensor representation.
        
        Args:
            sequence: A protein sequence as a string of amino acid one-letter codes
            
        Returns:
            torch.Tensor: A tensor of shape [embedding_dim] representing the protein
        """
        features = self.compute_protein_features(sequence)
        return self._features_to_tensor(features)
    
    def encode_batch(self, batch_data: List[str]) -> torch.Tensor:
        """
        Encode a batch of protein sequences into tensor representations.
        
        Args:
            batch_data: A list of protein sequences
            
        Returns:
            torch.Tensor: A tensor of shape [batch_size, embedding_dim]
        """
        features_list = [self.compute_protein_features(seq) for seq in batch_data]
        tensors = [self._features_to_tensor(features) for features in features_list]
        return torch.stack(tensors)
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def to(self, device):
        # This encoder doesn't have parameters that need to be moved to a device
        # But we implement this method for compatibility with other encoders
        return self