"This was created in combination of GPT4o, https://blog.dataiku.com/graph-neural-networks-part-three and Maxym's Glycan GNN"

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import NNConv, global_mean_pool
from typing import List

from ...base.encoders import GlycanEncoder

class ACONNEncoderV2(GlycanEncoder):
    def __init__(self, max_nodes=512, embedding_dim=32, hidden_dim=32, dropout_prob=0.2):
        super().__init__()

        self.max_nodes = max_nodes
        self._embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Regularization
        self.dropout = nn.Dropout(p=dropout_prob)

        # Embedding for atoms (nodes)
        self.atom_embedding = nn.Embedding(max_nodes, embedding_dim)

        # Edge encoder: maps edge_attr → message weights
        self.edge_attr_dim = 6  # 4 bond types + conjugation + ring
        edge_mlp = nn.Sequential(
            nn.Linear(self.edge_attr_dim, hidden_dim * embedding_dim),
            nn.ReLU()
        )

        self.conv1 = NNConv(embedding_dim, hidden_dim, edge_mlp, aggr='mean')

        edge_mlp2 = nn.Sequential(
            nn.Linear(self.edge_attr_dim, hidden_dim * hidden_dim),
            nn.ReLU()
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, edge_mlp2, aggr='mean')

    def smiles_to_graph(self, smiles: str) -> Data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        num_atoms = mol.GetNumAtoms()
        if num_atoms > self.max_nodes:
            raise ValueError(f"Number of atoms ({num_atoms}) exceeds max_nodes ({self.max_nodes})")

        node_ids = torch.arange(num_atoms, dtype=torch.long)
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_type = bond.GetBondType()
            is_conjugated = bond.GetIsConjugated()
            is_in_ring = bond.IsInRing()

            bond_type_encoding = {
                Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0],
                Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0],
                Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0],
                Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1],
            }.get(bond_type, [0, 0, 0, 0])

            features = torch.tensor(
                bond_type_encoding + [int(is_conjugated), int(is_in_ring)],
                dtype=torch.float
            )

            for (a, b) in [(i, j), (j, i)]:
                edge_index.append([a, b])
                edge_attr.append(features)

        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.stack(edge_attr, dim=0)

        return Data(x=node_ids, edge_index=edge_index, edge_attr=edge_attr)

    def forward(self, smiles: str, device: torch.device) -> torch.Tensor:
        data = self.smiles_to_graph(smiles).to(device)

        x = self.atom_embedding(data.x)
        x = self.conv1(x, data.edge_index, data.edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        graph_embedding = global_mean_pool(x, batch=None)
        return graph_embedding.unsqueeze(0)

    def encode_smiles(self, smiles: str, device: torch.device) -> torch.Tensor:
        return self.forward(smiles, device)

    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        data_list = [self.smiles_to_graph(s) for s in batch_data]
        batch = Batch.from_data_list(data_list).to(device)

        x = self.atom_embedding(batch.x)
        x = self.conv1(x, batch.edge_index, batch.edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        graph_embeddings = global_mean_pool(x, batch.batch)
        return graph_embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim