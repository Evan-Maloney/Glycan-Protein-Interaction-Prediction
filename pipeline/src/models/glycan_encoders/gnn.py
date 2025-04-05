# Soource: https://blog.dataiku.com/graph-neural-networks-part-three
# With help from Claude 3.7 to convert GNN to our use case

import torch
import torch.nn.functional as F

from typing import List
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from ...base.encoders import GlycanEncoder

from rdkit import Chem
#from rdkit.Chem import AllChem


class GNNGlycanEncoder(GlycanEncoder):
    def __init__(self, embedding_dim: int = 256, hidden_channels: int = 128):
        super().__init__()
        
        # Node features (9-dimensional as shown in the figure)
        self.node_features = [
            'atomic_num',      # Atomic number
            'chirality',       # Chirality (important for glycans)
            'degree',          # Degree (number of bonds)
            'formal_charge',   # Formal charge
            'numH',            # Number of hydrogens
            'number_radical_e', # Number of radical electrons
            'hybridization',   # Hybridization type
            'is_aromatic',     # Is the atom aromatic (boolean)
            'is_in_ring'       # Is the atom in a ring (boolean)
        ]
        
        # Edge features (3-dimensional as shown in the figure)
        self.edge_features = [
            'bond_type',         # Type of bond (single, double, etc.)
            'stereo_configuration', # Stereo configuration
            'is_conjugated'      # Is the bond conjugated (boolean)
        ]
        
        # Define normalization parameters (to be populated during preprocessing)
        self.scalers = {}
        
        # Define GNN layers
        self.conv1 = GCNConv(9, hidden_channels//2)  # 9 is the expanded node features
        self.conv2 = GCNConv(hidden_channels//2, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels*2)
        
        # Output layer
        self.linear = torch.nn.Linear(hidden_channels*2, embedding_dim)
        
        self._embedding_dim = embedding_dim
    
    def _get_atom_features(self, atom) -> List[float]:
        """Extract atom features according to the predefined list"""
        features = []
        
        # atomic_num
        features.append(atom.GetAtomicNum())
        
        # chirality
        chirality_type = int(atom.GetChiralTag())
        features.append(chirality_type)
        
        # degree
        features.append(atom.GetDegree())
        
        # formal_charge
        features.append(atom.GetFormalCharge())
        
        # numH
        features.append(atom.GetTotalNumHs())
        
        # number_radical_e
        features.append(atom.GetNumRadicalElectrons())
        
        # hybridization
        hybridization_type = int(atom.GetHybridization())
        features.append(hybridization_type)
        
        # is_aromatic
        features.append(1 if atom.GetIsAromatic() else 0)
        
        # is_in_ring
        features.append(1 if atom.IsInRing() else 0)
        
        return features
    
    def _get_bond_features(self, bond) -> List[float]:
        """Extract bond features according to the predefined list"""
        features = []
        
        # bond_type
        bond_type = int(bond.GetBondType())
        features.append(bond_type)
        
        # stereo_configuration
        stereo = int(bond.GetStereo())
        features.append(stereo)
        
        # is_conjugated
        features.append(1 if bond.GetIsConjugated() else 0)
        
        return features
    
    def _mol_to_graph_data(self, mol) -> Data:
        """Convert an RDKit molecule to a PyTorch Geometric Data object"""
        # Get atom features
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(self._get_atom_features(atom))
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Get edge indices and features
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # Add reverse edge for undirected graph
            
            features = self._get_bond_features(bond)
            edge_features.append(features)
            edge_features.append(features)  # Duplicate for reverse edge
        
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Handle molecules with no bonds (rare case)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float)
        
        # Create and normalize features
        x_norm = self._normalize_node_features(x)
        edge_attr_norm = self._normalize_edge_features(edge_attr)
        
        # Create the PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            x_norm=x_norm,
            edge_attr_norm=edge_attr_norm
        )
        
        return data
    
    def _normalize_node_features(self, x):
        """Apply normalization and one-hot encoding to node features"""
        # This is a placeholder - in production you'd use the scalers and encoding logic
        # from your preprocessing code
        # For simplicity, we're just returning the raw features
        return x
    
    def _normalize_edge_features(self, edge_attr):
        """Apply normalization and one-hot encoding to edge features"""
        # This is a placeholder - in production you'd use the scalers and encoding logic
        # from your preprocessing code
        # For simplicity, we're just returning the raw features
        return edge_attr
    
    def forward(self, data):
        """Process a batch of molecular graphs through the GNN"""
        x, edge_index, batch = data.x_norm, data.edge_index, data.batch
        
        # Apply GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Global pooling to get a graph-level representation
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5)
        
        # Final projection to embedding dimension
        x = self.linear(x)
        
        return x
    
    
    def encode_iupac(self, iupac_str: str, device: torch.device) -> torch.Tensor:
        """aaaaaa"""
        pass
    
    def encode_smiles(self, smiles: str, device: torch.device) -> torch.Tensor:
        """Convert a SMILES string to a graph embedding"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")
        
        # Optionally add hydrogen atoms
        mol = Chem.AddHs(mol)
        
        # Convert to a graph data object
        data = self._mol_to_graph_data(mol)
        
        # Create a batch with just this single molecule
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        
        # Move to device
        data = data.to(device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.forward(data)
        
        return embedding
    
    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        """Convert a batch of SMILES strings to graph embeddings"""
        # Process each molecule individually
        batch_embeddings = []
        for smiles in batch_data:
            embedding = self.encode_smiles(smiles, device)
            batch_embeddings.append(embedding)
        #for iupac in batch_data:
            #embedding = self.encode_iupac(iupac, device)
            #batch_embeddings.append(embedding)
        
        # Stack all embeddings
        batch = torch.cat(batch_embeddings, dim=0)
        
        return batch
    
    def preprocess_dataset(self, smiles_list: List[str]):
        """Precompute normalization parameters for the dataset"""
        # Convert all molecules to graphs and collect statistics
        all_node_features = []
        all_edge_features = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Collect node features
            for atom in mol.GetAtoms():
                features = self._get_atom_features(atom)
                all_node_features.append(features)
            
            # Collect edge features
            for bond in mol.GetBonds():
                features = self._get_bond_features(bond)
                all_edge_features.append(features)
        
        # Convert to tensors
        all_node_features = torch.tensor(all_node_features, dtype=torch.float)
        all_edge_features = torch.tensor(all_edge_features, dtype=torch.float)
        
        # Compute normalization parameters
        for i, feature_name in enumerate(self.node_features):
            feature_values = all_node_features[:, i]
            self.scalers[feature_name] = {
                'min': float(feature_values.min()),
                'max': float(feature_values.max()),
                'mean': float(feature_values.mean()),
                'std': float(feature_values.std())
            }
        
        for i, feature_name in enumerate(self.edge_features):
            feature_values = all_edge_features[:, i]
            self.scalers[feature_name] = {
                'min': float(feature_values.min()),
                'max': float(feature_values.max()),
                'mean': float(feature_values.mean()),
                'std': float(feature_values.std())
            }
        
        return self.scalers
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim