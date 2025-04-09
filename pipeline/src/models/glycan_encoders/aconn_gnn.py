"This was created in combination of GPT4o, https://blog.dataiku.com/graph-neural-networks-part-three and Maxym's Glycan GNN"

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from ...base.encoders import GlycanEncoder

class ACONNEncoder(GlycanEncoder):
    def __init__(self, max_nodes=512, embedding_dim=32, hidden_dim=32, dropout_prob=0.2):
        super().__init__()

        self.max_nodes = max_nodes
        self._embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Learnable embeddings for atomic positions (not types)
        self.atom_embedding = nn.Embedding(max_nodes, embedding_dim)

        # GNN layers (GCN-based)
        self.gcn1 = GCNConv(embedding_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Regularization
        self.dropout = nn.Dropout(p=dropout_prob)

    def smiles_to_graph(self, smiles):
        """Converts a SMILES string to a PyTorch Geometric graph."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        num_atoms = mol.GetNumAtoms()
        if num_atoms > self.max_nodes:
            raise ValueError(f"Number of atoms ({num_atoms}) exceeds max_nodes ({self.max_nodes})")

        node_ids = torch.arange(num_atoms, dtype=torch.long)

        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index += [[i, j], [j, i]]  # Undirected edges

        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        return Data(x=node_ids, edge_index=edge_index)

    def forward(self, smiles):
        """Takes a SMILES string and returns a glycan graph embedding."""
        data = self.smiles_to_graph(smiles)

        # Atom position embeddings
        x = self.atom_embedding(data.x)

        # GCN layers with ReLU and dropout
        x = self.gcn1(x, data.edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.gcn2(x, data.edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Graph pooling to get fixed-size embedding
        x = global_mean_pool(x, batch=None)  # shape: [hidden_dim]
        return x.unsqueeze(0)
    
    def encode_smiles(self, smiles: str, device: torch.device):
        """
        Wrapper for compatibility with pipeline API.
        """
        return self.forward(smiles, device)
    
    def encode_iupac(self, iupacs: str, device: torch.device) -> torch.Tensor:
        pass

    def encode_batch(self, smiles_list, device):
        """
        Encode a list of SMILES strings into a batch of graph embeddings.
        All tensors are moved to the given device.
        
        Args:
            smiles_list (List[str]): List of SMILES strings.
            device (torch.device): Target device (e.g., 'cuda' or 'cpu').
        
        Returns:
            torch.Tensor: Graph embeddings [batch_size, hidden_dim]
        """
        # Convert SMILES strings to graph Data objects
        data_list = [self.smiles_to_graph(smiles) for smiles in smiles_list]

        # Create batched graph and move to device
        batch = Batch.from_data_list(data_list).to(device)

        # Apply atom embeddings
        x = self.atom_embedding(batch.x.to(device))

        # GCN message passing
        x = self.gcn1(x, batch.edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.gcn2(x, batch.edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Global graph pooling (mean over atoms in each molecule)
        graph_embeddings = global_mean_pool(x, batch.batch)  # [batch_size, hidden_dim]

        return graph_embeddings
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim