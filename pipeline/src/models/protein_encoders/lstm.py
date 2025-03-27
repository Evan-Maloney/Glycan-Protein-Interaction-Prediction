import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from ...base.encoders import ProteinEncoder


class LSTMProteinEncoder(ProteinEncoder):
    def __init__(self, embedding_dim=16, hidden_dim=32, output_dim=12, dropout=0.2):
        super().__init__()
        self._embedding_dim = output_dim

        # Amino acid vocabulary: 20 canonical AAs + padding
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.pad_token = "_"
        self.vocab = {aa: i + 1 for i, aa in enumerate(self.amino_acids)}
        self.vocab[self.pad_token] = 0
        self.vocab_size = len(self.vocab)

        # Embedding and LSTM
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Final projection to output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        """
        Core method: encode a batch of amino acid sequences.

        Args:
            sequences (List[str]): Amino acid sequences
            device (torch.device): Device to run computation on

        Returns:
            torch.Tensor: [batch_size, output_dim]
        """
        # Tokenize sequences to integer IDs
        tokenized = [torch.tensor(
            [self.vocab.get(aa, 0) for aa in seq],
            dtype=torch.long
        ) for seq in sequences]

        max_len = max(len(seq) for seq in tokenized)
        padded = [F.pad(seq, (0, max_len - len(seq))) for seq in tokenized]
        batch_tensor = torch.stack(padded).to(device)  # [B, L]

        # Embedding → LSTM → mean pooling → projection
        x = self.embedding(batch_tensor)      # [B, L, E]
        out, _ = self.lstm(x)                 # [B, L, 2H]
        pooled = out.mean(dim=1)              # [B, 2H]
        return self.output_proj(pooled)       # [B, output_dim]

    def encode_sequence(self, sequence: str, device: torch.device) -> torch.Tensor:
        """
        Encode a single amino acid sequence to a tensor.

        Args:
            sequence (str): Amino acid sequence

        Returns:
            torch.Tensor: [1, output_dim]
        """
        return self.forward([sequence], device)

    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        """
        Encode a list of amino acid sequences to a batch tensor.

        Args:
            batch_data (List[str]): List of amino acid sequences

        Returns:
            torch.Tensor: [batch_size, output_dim]
        """
        return self.forward(batch_data, device)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
