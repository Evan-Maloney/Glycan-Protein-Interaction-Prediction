import torch
import torch.nn as nn
import torch.nn.functional as F
from ...base.predictors import BindingPredictor

class AttentionFusionBindingPredictor(BindingPredictor):
    def __init__(self, glycan_dim: int, protein_dim: int, hidden_dim: int = 128):
        super().__init__(glycan_dim, protein_dim)

        self.concentration_dim = 1
        self.total_dim = glycan_dim + protein_dim + self.concentration_dim

        # Learn attention weights over each modality (glycan, protein, concentration)
        self.attn_weights = nn.Parameter(torch.ones(3))  # Will be softmaxed

        # Feedforward MLP after attention fusion
        self.mlp = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self,
                glycan_encoding: torch.Tensor,
                protein_encoding: torch.Tensor,
                concentration: torch.Tensor) -> torch.Tensor:
        """
        glycan_encoding:     [B, glycan_dim]
        protein_encoding:    [B, protein_dim]
        concentration:       [B, 1]
        """

        # Concatenate raw inputs
        x_concat = torch.cat([glycan_encoding, protein_encoding, concentration], dim=-1)

        # Split back into modalities (to apply attention weights)
        B = glycan_encoding.shape[0]
        splits = [self.glycan_dim, self.protein_dim, self.concentration_dim]
        parts = torch.split(x_concat, splits, dim=-1)  # list of tensors

        # Softmax attention weights
        attn = F.softmax(self.attn_weights, dim=0)

        # Weighted fusion
        weighted_parts = [attn[i] * parts[i] for i in range(3)]
        fused = torch.cat(weighted_parts, dim=-1)  # [B, total_dim]

        # Final prediction
        output = self.mlp(fused)
        return torch.sigmoid(output)  # Binding probability in [0, 1]
