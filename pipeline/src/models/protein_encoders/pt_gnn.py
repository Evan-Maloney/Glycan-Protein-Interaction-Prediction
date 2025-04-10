import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from ...base.encoders import ProteinEncoder

"This was created in combination of Claude 3.7, https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial and Maxym's Glycan GNN"

class AdvancedGNNProteinEncoder(nn.Module):
    """
    Advanced Graph Neural Network-based Protein Encoder that incorporates:
    - Rich amino acid feature representation
    - Flexible graph structures (sequential, predicted contacts)
    - Attention-based message passing
    - Multiple readout functions
    """
    def __init__(self, 
                 embedding_dim: int = 256, 
                 hidden_channels: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 use_attention: bool = True,
                 readout_mode: str = 'mean'):
        """
        Initialize the advanced GNN protein encoder.
        
        Args:
            embedding_dim: Final embedding dimension
            hidden_channels: Size of hidden layers in GNN
            num_layers: Number of GNN layers
            dropout: Dropout probability
            use_attention: Whether to use attention-based message passing
            readout_mode: Method for graph-level pooling ('mean', 'sum', 'max', 'mean+max')
        """
        super().__init__()
        self._embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.readout_mode = readout_mode
        
        # Feature dimensions
        self.aa_embedding_dim = 20  # One-hot encoding of amino acids
        self.physicochemical_dim = 12  # Various amino acid properties
        self.position_embedding_dim = 16  # Positional encoding
        
        # Total node feature dimension
        node_feature_dim = self.aa_embedding_dim + self.physicochemical_dim + self.position_embedding_dim
        
        # Amino acid mappings
        self.aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.default_idx = len(self.aa_to_idx)  # For unknown amino acids
        
        # Feature initialization layers
        self.position_embedding = nn.Embedding(1000, self.position_embedding_dim).cuda()  # Max sequence length of 1000
        
        # Physicochemical property mappings (pre-computed)
        self.aa_properties = self._initialize_aa_properties()
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer takes the combined node features
        if use_attention:
            self.convs.append(GATConv(node_feature_dim, hidden_channels, heads=4, concat=False))
        else:
            self.convs.append(GCNConv(node_feature_dim, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Additional layers
        for _ in range(num_layers - 1):
            if use_attention:
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output projections depend on readout mode
        output_dim = hidden_channels if 'mean+max' not in readout_mode else hidden_channels * 2
        self.projection = nn.Sequential(
            nn.Linear(output_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, embedding_dim)
        )
        
    def _initialize_aa_properties(self) -> Dict[str, torch.Tensor]:
        """Initialize physicochemical properties for each amino acid"""
        properties = {}
        
        # These values are based on common AA properties: 
        # hydrophobicity, charge, size, polarity, etc.
        
        # Define key properties for each amino acid (normalized)
        # Format: [hydrophobicity, charge, size, polarity, aromaticity, 
        #          h-bond donor, h-bond acceptor, pKa, pI, flexibility,
        #          reactivity, glycosylation_site]
        
        properties['A'] = torch.tensor([0.7, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.1, 0.0])
        properties['C'] = torch.tensor([0.8, 0.0, 0.2, 0.1, 0.0, 0.5, 0.0, 0.9, 0.4, 0.2, 0.9, 0.0])
        properties['D'] = torch.tensor([0.3, -1.0, 0.3, 0.9, 0.0, 0.0, 1.0, 0.1, 0.3, 0.5, 0.4, 0.0])
        properties['E'] = torch.tensor([0.4, -1.0, 0.4, 0.8, 0.0, 0.0, 1.0, 0.2, 0.3, 0.5, 0.3, 0.0])
        properties['F'] = torch.tensor([0.9, 0.0, 0.6, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.2, 0.0])
        properties['G'] = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.2, 0.0])
        properties['H'] = torch.tensor([0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.6, 0.7, 0.3, 0.6, 0.0])
        properties['I'] = torch.tensor([1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1, 0.1, 0.0])
        properties['K'] = torch.tensor([0.3, 1.0, 0.6, 0.8, 0.0, 0.5, 0.0, 1.0, 0.9, 0.5, 0.3, 0.0])
        properties['L'] = torch.tensor([0.9, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.1, 0.0])
        properties['M'] = torch.tensor([0.7, 0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.2, 0.0])
        properties['N'] = torch.tensor([0.3, 0.0, 0.3, 0.8, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.3, 1.0])
        properties['P'] = torch.tensor([0.5, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.2, 0.0])
        properties['Q'] = torch.tensor([0.4, 0.0, 0.4, 0.7, 0.0, 0.5, 0.5, 0.0, 0.5, 0.4, 0.2, 0.0])
        properties['R'] = torch.tensor([0.2, 1.0, 0.7, 0.9, 0.0, 0.5, 0.0, 0.5, 1.0, 0.4, 0.3, 0.0])
        properties['S'] = torch.tensor([0.4, 0.0, 0.2, 0.6, 0.0, 0.5, 0.5, 0.0, 0.5, 0.6, 0.2, 0.5])
        properties['T'] = torch.tensor([0.5, 0.0, 0.3, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.4, 0.2, 0.5])
        properties['V'] = torch.tensor([0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.1, 0.0])
        properties['W'] = torch.tensor([0.6, 0.0, 0.8, 0.1, 1.0, 0.5, 0.0, 0.0, 0.5, 0.1, 0.2, 0.0])
        properties['Y'] = torch.tensor([0.7, 0.0, 0.7, 0.4, 0.8, 0.5, 0.5, 0.3, 0.5, 0.2, 0.3, 0.0])
        
        # Default for unknown amino acids (average values)
        properties['X'] = torch.mean(torch.stack([prop for prop in properties.values()]), dim=0)
        
        return properties
        
    def _one_hot_encode_aa(self, aa: str) -> torch.Tensor:
        """One-hot encode an amino acid"""
        idx = self.aa_to_idx.get(aa, self.default_idx)
        one_hot = torch.zeros(self.aa_embedding_dim).cuda()
        if idx < self.aa_embedding_dim:
            one_hot[idx] = 1.0
        return one_hot.cuda()
    
    def _get_aa_properties(self, aa: str) -> torch.Tensor:
        """Get physicochemical properties for an amino acid"""
        return self.aa_properties.get(aa, self.aa_properties['X']).cuda()
    
    def _sequence_to_graph(self, 
                          sequence: str, 
                          contact_map: Optional[Union[torch.Tensor, List, np.ndarray, None]] = None,
                          distance_threshold: float = 8.0) -> Data:
        """
        Convert a protein sequence to a graph representation.
        
        Args:
            sequence: Amino acid sequence
            contact_map: Optional tensor of pairwise distances/contacts
            distance_threshold: Threshold for considering residues in contact
            
        Returns:
            PyTorch Geometric Data object
        """
        # Node features: combine one-hot encoding, properties, and position
        x = []
        for i, aa in enumerate(sequence):
            if aa not in self.aa_to_idx and aa != 'X':
                aa = 'X'  # Use default for unknown amino acids
                
            # Combine features
            #aa = aa.cuda()
            one_hot = self._one_hot_encode_aa(aa).cuda()
            properties = self._get_aa_properties(aa).cuda()
            position = self.position_embedding(torch.tensor([min(i, 999)]).cuda()).cuda()
            
            # Concatenate all features
            features = torch.cat([one_hot, properties, position.squeeze(0)]).cuda()
            x.append(features)
            
        # Create node features tensor
        x = torch.stack(x).cuda()
        
        # Create edge index
        edge_index = []
        
        # Add sequential connections (each AA connected to neighbors within window)
        window_size = 3  # Connect each AA to this many neighbors in each direction
        for i in range(len(sequence)):
            # Connect to previous AAs within window
            for w in range(1, window_size + 1):
                if i - w >= 0:
                    edge_index.append([i-w, i])
                    edge_index.append([i, i-w])  # Bidirectional
            
            # Connect to next AAs within window
            for w in range(1, window_size + 1):
                if i + w < len(sequence):
                    edge_index.append([i, i+w])
                    edge_index.append([i+w, i])  # Bidirectional
        
        # Add contacts from contact map if provided
        if contact_map is not None:
            try:
                # Convert to tensor if not already
                if not isinstance(contact_map, torch.Tensor):
                    if isinstance(contact_map, np.ndarray):
                        contact_map = torch.tensor(contact_map).cuda()
                    elif isinstance(contact_map, list):
                        contact_map = torch.tensor(contact_map).cuda()
                
                # Only use contact map if it's now a tensor with the right shape
                if isinstance(contact_map, torch.Tensor) and contact_map.dim() == 2:
                    for i in range(len(sequence)):
                        for j in range(i + window_size + 1, min(len(sequence), contact_map.shape[0])):
                            # Check dimensions to avoid index errors
                            if i < contact_map.shape[0] and j < contact_map.shape[1]:
                                if contact_map[i, j] <= distance_threshold:
                                    edge_index.append([i, j])
                                    edge_index.append([j, i])  # Bidirectional
            except Exception as e:
                # If we encounter any error with the contact map, just ignore it
                print(f"Warning: Could not use contact map: {e}")
        
        # Create edge index tensor
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            # Handle case with no edges (very short sequence)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_index = edge_index.cuda()
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)
        return data.cuda()
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Process protein graph through the GNN.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Protein embedding tensor
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GNN layers with residual connections
        for i in range(self.num_layers):
            identity = x.cuda()
            x = self.convs[i](x, edge_index.cuda())
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)#, training=self.training)
            
            # Add residual connection if dimensions match
            if i > 0 and x.size(-1) == identity.size(-1):
                x = x + identity
        
        # Different pooling strategies
        if self.readout_mode == 'mean':
            x = global_mean_pool(x, batch)
        elif self.readout_mode == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout_mode == 'max':
            # Manual implementation of max pooling
            x_max, _ = global_max_pool(x, batch, dim=0)
            x = x_max
        elif self.readout_mode == 'mean+max':
            x_mean = global_mean_pool(x, batch)
            # Manual implementation of max pooling
            x_max, _ = global_max_pool(x, batch, dim=0)
            x = torch.cat([x_mean, x_max], dim=1)
        
        # Final projection
        x = self.projection(x).cuda()
        
        return x
    
    def encode_sequence(self, 
                         sequence: str, 
                         device: Optional[torch.device] = None,
                         contact_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a single protein sequence.
        
        Args:
            sequence: Amino acid sequence
            contact_map: Optional contact map for the protein
            
        Returns:
            Embedding tensor
        """
        # Convert sequence to graph
        data = self._sequence_to_graph(sequence, contact_map).to(device)
        
        # Add batch dimension for single sequence
        data.batch = torch.zeros(len(sequence), dtype=torch.long)
        
        # Move to device if specified
        if device is not None:
            data = data.to(device)
        
        # Forward pass
        with torch.no_grad():
            embedding = self.forward(data)
            
        return embedding
    
    def encode_batch(self, 
                     batch_data: List[str],
                     device: torch.device = None,
                     contact_maps: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Encode a batch of protein sequences.
        
        Args:
            batch_data: List of amino acid sequences
            device: Device to place tensors on
            contact_maps: Optional list of contact maps for each protein
            
        Returns:
            Batch of embedding tensors
        """
        print('deviceee,', device)
        #batch_data.to(device)
        # Create a list of Data objects
        data_list = []
        count = 0
        for sequence in batch_data:
            count += 1
            print(f'encode batch progress: {count}/{len(batch_data)}')
            # Don't use contact maps for now to avoid the error
            #sequence = sequence.cuda()
            data = self._sequence_to_graph(sequence, None).cuda()

            data_list.append(data)
            
        # Create a batch from the list
        batch = Batch.from_data_list(data_list)
        
        # Move to device if specified
        #if device is not None:
        batch = batch.cuda()
        
        # Forward pass
        with torch.no_grad():
            embeddings = self.forward(batch)
            
        return embeddings
    
    def predict_secondary_structure(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Predict secondary structure probabilities (helix, sheet, coil)
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of secondary structure probabilities
        """
        # This would require a separate prediction head
        # Here we use Biopython as a placeholder
        try:
            analysis = ProteinAnalysis(sequence)
            helix, turn, sheet = analysis.secondary_structure_fraction()
            
            # Convert to tensor format that could come from a model
            ss_pred = {
                'helix': torch.tensor([helix] * len(sequence)),
                'sheet': torch.tensor([sheet] * len(sequence)),
                'coil': torch.tensor([turn] * len(sequence))
            }
            return ss_pred
        except:
            # Default values if analysis fails
            return {
                'helix': torch.zeros(len(sequence)),
                'sheet': torch.zeros(len(sequence)),
                'coil': torch.ones(len(sequence))
            }
    
    def estimate_contact_map(self, sequence: str) -> torch.Tensor:
        """
        Estimate a contact map based on amino acid properties and sequential distance.
        This is a placeholder - ideally a dedicated contact prediction model would be used.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Estimated contact map (distances between residues)
        """
        seq_len = len(sequence)
        contact_map = torch.ones(seq_len, seq_len) * 100  # Initialize with large distances
        
        # Set sequential distances
        for i in range(seq_len):
            for j in range(seq_len):
                # Sequential distance penalty
                contact_map[i, j] = min(contact_map[i, j], abs(i - j) * 3.8)
                
                # Reduce distance for hydrophobic interactions
                aa_i = sequence[i] if sequence[i] in self.aa_to_idx else 'X'
                aa_j = sequence[j] if sequence[j] in self.aa_to_idx else 'X'
                hydrophobicity_i = self.aa_properties[aa_i][0]
                hydrophobicity_j = self.aa_properties[aa_j][0]
                
                # Hydrophobic residues tend to cluster
                if hydrophobicity_i > 0.7 and hydrophobicity_j > 0.7:
                    contact_map[i, j] = min(contact_map[i, j], 8.0 + abs(i - j) * 0.5)
                
                # Ionic interactions between charged residues
                charge_i = self.aa_properties[aa_i][1]
                charge_j = self.aa_properties[aa_j][1]
                if abs(i - j) > 4 and charge_i * charge_j < 0:  # Opposite charges attract
                    contact_map[i, j] = min(contact_map[i, j], 10.0)
                    
        return contact_map
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim