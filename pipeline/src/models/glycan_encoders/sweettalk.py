import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from ...base.encoders import GlycanEncoder


class SweetTalkGlycanEncoder(GlycanEncoder):
    def __init__(self, model_path: str, char_vocab_path: str = None, glycoword_vocab_path: str = None, embedding_dim: int = 128):
        """
        Initialize the SweetTalk glycan encoder.
        
        Args:
            model_path: Path to the pre-trained SweetTalk model (.pt file)
            char_vocab_path: Path to a file containing the character vocabulary (optional)
            glycoword_vocab_path: Path to a file containing the glycoword vocabulary (optional)
            embedding_dim: Desired dimensionality of the output embeddings
        """
        super().__init__()
        
        # Load the pre-trained SweetTalk model
        self.model = torch.load(model_path)
        self.model.eval()  # Set to evaluation mode
        
        # Extract the embedding layer which contains the learned representations
        self.encoder = self.model.encoder
        
        # Get the character-level embeddings
        self.char_embeddings = self.encoder.weight.data.cpu().numpy()
        
        # Initialize vocabularies
        self.char_vocab = self._load_vocab(char_vocab_path) if char_vocab_path else self._initialize_char_vocab()
        self.glycoword_vocab = self._load_vocab(glycoword_vocab_path) if glycoword_vocab_path else self._initialize_glycoword_vocab()
        
        # Create glycoword embeddings by averaging character embeddings
        self.glycoword_embeddings = self._create_glycoword_embeddings()
        
        self._embedding_dim = embedding_dim
        
        # Optional projection layer if you want to change the embedding dimension
        if embedding_dim != 128:  # SweetTalk uses 128-dimensional embeddings
            self.projection = nn.Linear(128, embedding_dim)
        else:
            self.projection = nn.Identity()
    
    def _load_vocab(self, path: str) -> Dict[str, int]:
        """Load vocabulary from a file."""
        vocab = {}
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                vocab[line.strip()] = i
        return vocab
    
    def _initialize_char_vocab(self) -> Dict[str, int]:
        """
        Initialize the character-level vocabulary.
        This should match the 'lib_char' from SweetTalk.
        """
        # loading training set to create mappings from there (might )
        return {
            'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
            # ... other characters
        }
    
    def _initialize_glycoword_vocab(self) -> Dict[str, int]:
        """
        Initialize the glycoword-level vocabulary.
        This should match the 'lib_all_long' from SweetTalk.
        """
        # This is a placeholder - you need to get the actual glycoword vocabulary
        return {
            "Glc": 0,
            "Gal": 1,
            # ... other glycowords
        }
    
    def _create_glycoword_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Create glycoword embeddings by averaging character embeddings.
        This mirrors the approach used in the SweetTalk code.
        """
        glycoword_embeddings = {}
        
        for glycoword, idx in self.glycoword_vocab.items():
            if glycoword == "<PAD>":
                # Padding embedding is all zeros
                glycoword_embeddings[glycoword] = np.zeros(128)
                continue
                
            # Convert glycoword to character indices
            char_indices = [self.char_vocab[char] for char in glycoword if char in self.char_vocab]
            
            # Average the character embeddings
            if char_indices:
                embedding = np.mean(self.char_embeddings[char_indices], axis=0)
                glycoword_embeddings[glycoword] = embedding
            else:
                glycoword_embeddings[glycoword] = np.zeros(128)
        
        return glycoword_embeddings
    
    def motif_find(self, s: str) -> List[str]:
        """
        Converts a IUPAC-condensed glycan into a list of overlapping, asterisk-separated glycowords.
        """
        b = s.split('(')
        b = [k.split(')') for k in b]
        b = [item for sublist in b for item in sublist]
        b = [k.strip('[') for k in b]
        b = [k.strip(']') for k in b]
        b = [k.replace('[', '') for k in b]
        b = [k.replace(']', '') for k in b]
        b = ['*'.join(b[i:i+5]) for i in range(0, len(b)-4, 2)]
        return b
    
    def small_motif_find(self, s: str) -> str:
        """
        Converts a IUPAC-condensed glycan into an asterisk-separated string of glycoletters.
        """
        b = s.split('(')
        b = [k.split(')') for k in b]
        b = [item for sublist in b for item in sublist]
        b = [k.strip('[') for k in b]
        b = [k.strip(']') for k in b]
        b = [k.replace('[', '') for k in b]
        b = [k.replace(']', '') for k in b]
        b = '*'.join(b)
        return b
    
    def process_glycan(self, iupac: str) -> List[List[str]]:
        """
        Converts a glycan into a list of lists of glycowords.
        """
        motifs = self.motif_find(iupac)
        return [motif.split('*') for motif in motifs]
    
    def encode_iupac(self, iupac: str, device: torch.device) -> torch.Tensor:
        """
        Encode a single IUPAC glycan string to an embedding vector.
        """
        # Process the glycan into glycowords
        glycoword_lists = self.process_glycan(iupac)
        
        # If no glycowords were found, return zero embedding
        if not glycoword_lists:
            return torch.zeros(self._embedding_dim, device=device)
        
        # Get embeddings for each glycoword in each list
        list_embeddings = []
        for glycoword_list in glycoword_lists:
            word_embeddings = []
            for word in glycoword_list:
                if word in self.glycoword_embeddings:
                    word_embeddings.append(self.glycoword_embeddings[word])
                else:
                    # For unknown glycowords, create embedding by averaging character embeddings
                    char_indices = [self.char_vocab[char] for char in word if char in self.char_vocab]
                    if char_indices:
                        word_embeddings.append(np.mean(self.char_embeddings[char_indices], axis=0))
                    else:
                        word_embeddings.append(np.zeros(128))
            
            # Average the embeddings for this list of glycowords
            if word_embeddings:
                list_embeddings.append(np.mean(word_embeddings, axis=0))
        
        # Average all list embeddings to get the final embedding
        final_embedding = np.mean(list_embeddings, axis=0)
        
        # Convert to tensor and move to device
        embedding_tensor = torch.tensor(final_embedding, dtype=torch.float32).to(device)
        
        # Apply projection if needed
        projected = self.projection(embedding_tensor)
        
        return projected
    
    def encode_iupac_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        """
        Encode a batch of IUPAC glycan strings to embedding vectors.
        """
        batch_embeddings = []
        
        for iupac in batch_data:
            embedding = self.encode_iupac(iupac, device)
            batch_embeddings.append(embedding)
        
        # Stack embeddings into a batch
        batch_tensor = torch.stack(batch_embeddings, dim=0)
        
        return batch_tensor
    
    def encode_batch(self, batch_data: List[str], device: torch.device) -> torch.Tensor:
        """
        This method assumes batch_data contains IUPAC strings.
        If you need to handle SMILES, you would need to convert SMILES to IUPAC first.
        """
        return self.encode_iupac_batch(batch_data, device)
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        If x is already encoded, just pass it through.
        This is for compatibility with the encoder interface.
        """
        return x