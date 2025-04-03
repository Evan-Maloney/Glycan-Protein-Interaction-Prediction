# Reference: 
#   -https://www.youtube.com/watch?v=YYoXGYSbH3Q&t=1594s&ab_channel=ValenceLabs
#   -https://arxiv.org/pdf/2110.07875        
#   -https://arxiv.org/pdf/1704.01212
#   -https://arxiv.org/pdf/2010.02863
#   -https://arxiv.org/pdf/2106.03893
#   -https://arxiv.org/pdf/2205.12454
#   -Used GitHub copilot to help with the code

import torch
from typing import List
from ...base.encoders import GlycanEncoder

from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import BondStereo

import torch.nn as nn
import torch.nn.functional as F

class MPNNGlycanEncoder(GlycanEncoder):
    def __init__(self, embedding_dim: int = 128, pos_emb_dim: int = 2,
                 hidden_state_size: int = 128, n_layers: int = 3):
        super().__init__()

        # Node features (118 + 1 + 1 + 1 + 4 = 125 dimensions)
        self.node_features = [
            'atomic_num',
            'mass',
            'row',
            'column',
            'chirality',
        ]
        
        # Edge features (4 + 4 + 1 + 1 + 1 = 11 dimensions)
        self.edge_features = [
            'bond_type',
            'stero_configuration',
            'is_in_ring',
            'is_conjugated',
            'is_aromatic',
        ]
        
        self._embedding_dim = embedding_dim
        self.pos_emb_dim = pos_emb_dim
        self.hidden_state_size = hidden_state_size
        self.n_layers = n_layers

        # Assume base node features have 125 dimensions.
        # (For example: one-hot atomic number (118) + mass (1) + row (1) + column (1) + chirality one-hot (4))
        self.base_node_feature_dim = 125
        # After concatenating positional embeddings, node feature dim becomes:
        self.node_feature_dim = self.base_node_feature_dim + self.pos_emb_dim + 2

        # Edge features dimension (example): 11.
        # (For example: bond type one-hot (4) + stereo configuration one-hot (4) + is_in_ring (1) + is_conjugated (1) + is_aromatic (1))
        self.edge_feature_dim = 11

         # Initial projection to hidden state.
        self.initial_linear = nn.Linear(self.node_feature_dim, self.hidden_state_size)

        # Message passing function
        self.f_message = nn.Sequential(
            nn.Linear(self.hidden_state_size + self.edge_feature_dim, self.hidden_state_size),
            nn.ReLU(),
            nn.Linear(self.hidden_state_size, self.hidden_state_size)
        )
        
        # Update function
        self.f_update = nn.Sequential(
            nn.Linear(2 * self.hidden_state_size, self.hidden_state_size),
            nn.ReLU(),
            nn.Linear(self.hidden_state_size, self.hidden_state_size)
        )

        # Final readout projection.
        self.f_readout = nn.Linear(self.hidden_state_size, self._embedding_dim)
        
    def _get_random_walk_stats(self, adj: torch.Tensor, k_steps: int = 6) -> torch.Tensor:
        """
        Compute a k-step random walk bias matrix R = T^k (with T = D^-1 * A),
        and then derive per-node statistics (mean and std) for each node.
        Returns a tensor of shape (N, 2) where the two columns are mean and std.
        """
        deg = torch.sum(adj, dim=1, keepdim=True) + 1e-6
        T = adj / deg
        R = T.clone()
        for _ in range(k_steps - 1):
            R = R @ T
        # For each node, compute mean and standard deviation across the row.
        r_mean = R.mean(dim=1, keepdim=True)  # (N, 1)
        r_std = R.std(dim=1, keepdim=True)    # (N, 1)
        return torch.cat([r_mean, r_std], dim=1)  # (N, 2)


    def _get_positional_embeddings(self, adj: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute k-dimensional positional embeddings using the Laplacian eigenvectors.
        adj: (N, N) adjacency matrix.
        Returns: Tensor of shape (N, k)
        """
        # Compute degree vector and construct degree matrix D.
        deg = torch.sum(adj, dim=1)
        D = torch.diag(deg)
        # Compute Laplacian: L = D - A.
        L = D - adj
        # Compute eigen-decomposition (eigenvalues in ascending order).
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        # Skip the first eigenvector (trivial constant vector) if possible.
        if k < L.size(0):
            pos_emb = eigenvectors[:, 1:k+1]
        else:
            pos_emb = eigenvectors[:, :k]
        return pos_emb
    
    def _one_hot_atomic_number(self, atom):
        one_hot = [0] * 118
        atomic_num = atom.GetAtomicNum()
        one_hot[atomic_num - 1] = 1
        return one_hot
    
    def _one_hot_chirality(self, atom):
        chiral_tag = atom.GetChiralTag()
        possible_tags = [
            ChiralType.CHI_UNSPECIFIED, 
            ChiralType.CHI_TETRAHEDRAL_CW, 
            ChiralType.CHI_TETRAHEDRAL_CCW, 
            ChiralType.CHI_OTHER,
        ]
        one_hot = [1 if chiral_tag == tag else 0 for tag in possible_tags]
        return one_hot

    def _one_hot_bond_type(self, bond):
        bond_type = bond.GetBondType()
        possible_types = [
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
        ]
        one_hot = [1 if bond_type == type else 0 for type in possible_types]
        return one_hot
    
    def _one_hot_stereo_configuration(self, bond):
        stereo = bond.GetStereo()
        possible_configurations = [
            BondStereo.STEREOANY,
            BondStereo.STEREOZ,
            BondStereo.STEREOE,
            BondStereo.STEREONONE,
        ]
        one_hot = [1 if stereo == config else 0 for config in possible_configurations]
        return one_hot
    
    def _get_atom_features(self, atom) -> List[float]:
        """Extract atom features according to the predefined list"""
        features = []

        # atomic number one hot encoding
        features += self._one_hot_atomic_number(atom)

        # atomic mass
        features.append(atom.GetMass())

        # row in periodic table / period
        features.append(element_row[atom.GetSymbol()])

        # column in periodic table / group
        features.append(element_col[atom.GetSymbol()])

        # chirality one hot encoding
        features += self._one_hot_chirality(atom)

        return features
    
    def _get_bond_features(self, bond) -> List[float]:
        """Extract bond features according to the predefined list"""
        features = []

        # bond type one hot encoding
        features += self._one_hot_bond_type(bond)

        # stereo configuration one hot encoding
        features += self._one_hot_stereo_configuration(bond)

        # is in ring
        features.append(bond.IsInRing())

        # is conjugated
        features.append(bond.GetIsConjugated())

        # is aromatic
        features.append(bond.GetIsAromatic())

        return features
    
    def _mol_to_graph_data(self, mol) -> dict:
        """
        Convert an RDKit molecule to a graph data dictionary containing:
          - x: node feature matrix (N x node_feature_dim)
          - adj: adjacency matrix (N x N)
          - edge_attr: edge feature tensor (N x N x edge_feature_dim)
          - batch: tensor indicating graph membership (for a single graph, all zeros)
        """
        # Build node features.
        node_features = [self._get_atom_features(atom) for atom in mol.GetAtoms()]
        x_raw = torch.tensor(node_features, dtype=torch.float)  # Shape: (N, node_feature_dim)
        N = x_raw.size(0)
        
        # Initialize dense adjacency and edge feature matrices.
        adj = torch.zeros((N, N), dtype=torch.float)
        edge_attr = torch.zeros((N, N, self.edge_feature_dim), dtype=torch.float)
        
        # Populate matrices for each bond.
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            bf = self._get_bond_features(bond)
            bf_tensor = torch.tensor(bf, dtype=torch.float)
            edge_attr[i, j] = bf_tensor
            edge_attr[j, i] = bf_tensor
        
        # Compute positional embeddings from the Laplacian.
        pos_emb = self._get_positional_embeddings(adj, self.pos_emb_dim)  # Shape: (N, pos_emb_dim)

        rw_stats = self._get_random_walk_stats(adj)  # Shape: (N, 2)
        # Concatenate positional embeddings to raw node features.
        x = torch.cat([x_raw, pos_emb, rw_stats], dim=1)  # Shape: (N, base_node_feature_dim + pos_emb_dim)
        
        # Optionally, apply normalization (here we pass features through).
        x_norm = self._normalize_node_features(x)
        edge_attr_norm = self._normalize_edge_features(edge_attr)
        
        # For a single graph, assign all nodes to batch 0.
        batch = torch.zeros(N, dtype=torch.long)
        
        data = {
            'x': x,
            'adj': adj,
            'edge_attr': edge_attr,
            'x_norm': x_norm,
            'edge_attr_norm': edge_attr_norm,
            'batch': batch
        }
        return data 
    
    def _normalize_node_features(self, x):
        """Placeholder for node feature normalization."""
        return x

    def _normalize_edge_features(self, edge_attr):
        """Placeholder for edge feature normalization."""
        return edge_attr
    
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
        data['batch'] = torch.zeros(data['x'].size(0), dtype=torch.long)

        # Move all tensor entries to device.
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
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
            #mol = Chem.AddHs(mol)
            
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
    
    def forward(self, data: dict) -> torch.Tensor:
        """
        Perform message passing to produce a graph-level embedding.
        data: Dictionary containing keys 'x_norm', 'adj', 'edge_attr_norm', 'batch'.
        """
        # Extract inputs.
        x = data['x_norm']        # (N, node_feature_dim) where node_feature_dim = base (125) + pos_emb_dim
        adj = data['adj']         # (N, N)
        edge_attr = data['edge_attr_norm']  # (N, N, edge_feature_dim)
        N = x.size(0)

        # Initial projection to hidden state.
        h = F.relu(self.initial_linear(x))  # (N, hidden_state_size)

        # Message passing iterations.
        for t in range(self.n_layers):
            h_neighbors = h.unsqueeze(0).expand(N, N, self.hidden_state_size)  # (N, N, hidden_state_size)
            msg_input = torch.cat([h_neighbors, edge_attr], dim=2)  # (N, N, hidden_state_size + edge_feature_dim)
            msg_input_flat = msg_input.view(-1, self.hidden_state_size + self.edge_feature_dim)
            messages_flat = self.f_message(msg_input_flat)  # (N*N, hidden_state_size)
            messages = messages_flat.view(N, N, self.hidden_state_size)  # (N, N, hidden_state_size)
            messages = messages * adj.unsqueeze(2)  # mask non-existent edges
            m = messages.sum(dim=1)  # aggregate messages by mean: (N, hidden_state_size)
            h = F.relu(self.f_update(torch.cat([h, m], dim=1)))  # update node states: (N, hidden_state_size)

        # Global mean pooling.
        graph_repr = h.sum(dim=0, keepdim=True)  # (1, hidden_state_size)
        out = self.f_readout(graph_repr)  # (1, embedding_dim)
        return out
    
# Hashmap for periodic table row (period) (used RdKit pt.GetRow())
element_row = {
    'H': 1,
    'He': 1,
    'Li': 2,
    'Be': 2,
    'B': 2,
    'C': 2,
    'N': 2,
    'O': 2,
    'F': 2,
    'Ne': 2,
    'Na': 3,
    'Mg': 3,
    'Al': 3,
    'Si': 3,
    'P': 3,
    'S': 3,
    'Cl': 3,
    'Ar': 3,
    'K': 4,
    'Ca': 4,
    'Sc': 4,
    'Ti': 4,
    'V': 4,
    'Cr': 4,
    'Mn': 4,
    'Fe': 4,
    'Co': 4,
    'Ni': 4,
    'Cu': 4,
    'Zn': 4,
    'Ga': 4,
    'Ge': 4,
    'As': 4,
    'Se': 4,
    'Br': 4,
    'Kr': 4,
    'Rb': 5,
    'Sr': 5,
    'Y': 5,
    'Zr': 5,
    'Nb': 5,
    'Mo': 5,
    'Tc': 5,
    'Ru': 5,
    'Rh': 5,
    'Pd': 5,
    'Ag': 5,
    'Cd': 5,
    'In': 5,
    'Sn': 5,
    'Sb': 5,
    'Te': 5,
    'I': 5,
    'Xe': 5,
    'Cs': 6,
    'Ba': 6,
    'La': 6,
    'Ce': 6,
    'Pr': 6,
    'Nd': 6,
    'Pm': 6,
    'Sm': 6,
    'Eu': 6,
    'Gd': 6,
    'Tb': 6,
    'Dy': 6,
    'Ho': 6,
    'Er': 6,
    'Tm': 6,
    'Yb': 6,
    'Lu': 6,
    'Hf': 6,
    'Ta': 6,
    'W': 6,
    'Re': 6,
    'Os': 6,
    'Ir': 6,
    'Pt': 6,
    'Au': 6,
    'Hg': 6,
    'Tl': 6,
    'Pb': 6,
    'Bi': 6,
    'Po': 6,
    'At': 6,
    'Rn': 6,
    'Fr': 7,
    'Ra': 7,
    'Ac': 7,
    'Th': 7,
    'Pa': 7,
    'U': 7,
    'Np': 7,
    'Pu': 7,
    'Am': 7,
    'Cm': 7,
    'Bk': 7,
    'Cf': 7,
    'Es': 7,
    'Fm': 7,
    'Md': 7,
    'No': 7,
    'Lr': 7,
    'Rf': 7,
    'Db': 7,
    'Sg': 7,
    'Bh': 7,
    'Hs': 7,
    'Mt': 7,
    'Ds': 7,
    'Rg': 7,
    'Cn': 7,
    'Nh': 7,
    'Fl': 7,
    'Mc': 7,
    'Lv': 7,
    'Ts': 7,
    'Og': 7,
}

# Hashmap for periodic table column (group) (manually entered)
element_col = {
    'H': 1,
    'He': 18,
    'Li': 1,
    'Be': 2,
    'B': 13,
    'C': 14,
    'N': 15,
    'O': 16,
    'F': 17,
    'Ne': 18,
    'Na': 1,
    'Mg': 2,
    'Al': 13,
    'Si': 14,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Ar': 18,
    'K': 1,
    'Ca': 2,
    'Sc': 3,
    'Ti': 4,
    'V': 5,
    'Cr': 6,
    'Mn': 7,
    'Fe': 8,
    'Co': 9,
    'Ni': 10,
    'Cu': 11,
    'Zn': 12,
    'Ga': 13,
    'Ge': 14,
    'As': 15,
    'Se': 16,
    'Br': 17,
    'Kr': 18,
    'Rb': 1,
    'Sr': 2,
    'Y': 3,
    'Zr': 4,
    'Nb': 5,
    'Mo': 6,
    'Tc': 7,
    'Ru': 8,
    'Rh': 9,
    'Pd': 10,
    'Ag': 11,
    'Cd': 12,
    'In': 13,
    'Sn': 14,
    'Sb': 15,
    'Te': 16,
    'I': 17,
    'Xe': 18,
    'Cs': 1,
    'Ba': 2,
    'La': 0,
    'Ce': 0,
    'Pr': 0,
    'Nd': 0,
    'Pm': 0,
    'Sm': 0,
    'Eu': 0,
    'Gd': 0,
    'Tb': 0,
    'Dy': 0,
    'Ho': 0,
    'Er': 0,
    'Tm': 0,
    'Yb': 0,
    'Lu': 3,
    'Hf': 4,
    'Ta': 5,
    'W': 6,
    'Re': 7,
    'Os': 8,
    'Ir': 9,
    'Pt': 10,
    'Au': 11,
    'Hg': 12,
    'Tl': 13,
    'Pb': 14,
    'Bi': 15,
    'Po': 16,
    'At': 17,
    'Rn': 18,
    'Fr': 1,
    'Ra': 2,
    'Ac': 0,
    'Th': 0,
    'Pa': 0,
    'U': 0,
    'Np': 0,
    'Pu': 0,
    'Am': 0,
    'Cm': 0,
    'Bk': 0,
    'Cf': 0,
    'Es': 0,
    'Fm': 0,
    'Md': 0,
    'No': 0,
    'Lr': 3,
    'Rf': 4,
    'Db': 5,
    'Sg': 6,
    'Bh': 7,
    'Hs': 8,
    'Mt': 9,
    'Ds': 10,
    'Rg': 11,
    'Cn': 12,
    'Nh': 13,
    'Fl': 14,
    'Mc': 15,
    'Lv': 16,
    'Ts': 17,
    'Og': 18,
}