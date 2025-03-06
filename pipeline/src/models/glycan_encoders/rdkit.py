import torch
from typing import List
from ...base.encoders import GlycanEncoder

from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors, Descriptors, GraphDescriptors


class RDKITGlycanEncoder(GlycanEncoder):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        self.features = [
            lambda mol: mol.GetNumHeavyAtoms(),
            lambda mol: Lipinski.NumRotatableBonds(mol),
            lambda mol: rdMolDescriptors.CalcNumRings(mol),
            lambda mol: rdMolDescriptors.CalcNumAromaticRings(mol),
            lambda mol: Descriptors.MolWt(mol),
            lambda mol: Descriptors.FractionCSP3(mol),
            lambda mol: Descriptors.TPSA(mol),
            lambda mol: len(Chem.FindMolChiralCenters(mol)),
            lambda mol: Lipinski.NumHDonors(mol),
            lambda mol: Lipinski.NumHAcceptors(mol),
            lambda mol: len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))),
            lambda mol: len(mol.GetSubstructMatches(Chem.MolFromSmarts('COC'))),
            lambda mol: len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)[O;H1]'))),
            lambda mol: GraphDescriptors.BalabanJ(mol),
            lambda mol: Descriptors.BertzCT(mol),
            lambda mol: Descriptors.MolLogP(mol),
            lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'),
            lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        ]
        
        self._embedding_dim = len(self.features)
        
        # Add a trainable linear transformation layer
        # self.features = [heavy_atoms, rot_bonds, num_rings, ..., num_carbon]
        # self.linear (num_features, embedding_dim=128)
        # Output_1 = w11 * heavy_atoms + w12 * rot_bonds + ... + w1n * num_carbon + bias_1
        # Output_2 = w21 * heavy_atoms + w22 * rot_bonds + ... + w2n * num_carbon + bias_2
        # And so on for 128 dimensions. --> [out_1, out_2, ..., out_128]
        # self.linear(18, 128) takes in vector of 18 features and converts it into a vector of size 128
        
        self.linear = torch.nn.Linear(len(self.features), self._embedding_dim)
    
    def encode_smiles(self, smiles: str) -> torch.Tensor:
        mol = Chem.MolFromSmiles(smiles)
        
        features = torch.tensor([feature_func(mol) for feature_func in self.features], dtype=torch.float32)
        return self.linear(features)
    
    def encode_batch(self, batch_data: List[str]) -> torch.Tensor:
        
        batch_features = [self.encode_smiles(smiles) for smiles in batch_data]
        # stack each feature as a row
        batch = torch.stack(batch_features, dim=0)
        
        return batch
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x