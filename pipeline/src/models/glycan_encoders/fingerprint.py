import torch
import numpy as np
from typing import List

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from ...base.encoders import GlycanEncoder  # adjust the import path as needed

class RDKITFingerprintEncoder(GlycanEncoder):
    def __init__(self, radius: int = 3, nBits: int = 2048):
        """
        Initializes the RDKit Fingerprint Encoder.

        Args:
            radius (int): The radius for the Morgan fingerprint.
            nBits (int): The length (number of bits) of the fingerprint.
        """
        super().__init__()
        self.radius = radius
        self.nBits = nBits
        self._embedding_dim = nBits

    def _fingerprint_from_smiles(self, smiles: str) -> torch.Tensor:
        """
        Computes the Morgan fingerprint for a given SMILES string using RDKit.

        Args:
            smiles (str): The SMILES string of the glycan.

        Returns:
            torch.Tensor: A 1D tensor of length `nBits` representing the fingerprint.
        """
        # Convert SMILES to an RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # If the SMILES string is invalid, return a zero vector.
            # Alternatively, you could raise an exception.
            return torch.zeros(self.nBits, dtype=torch.float)
        
        # Compute the Morgan fingerprint as a bit vector
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
        
        # Convert the fingerprint to a numpy array
        arr = np.zeros((self.nBits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        
        # Convert the numpy array to a torch tensor
        fingerprint_tensor = torch.tensor(arr, dtype=torch.float)
        return fingerprint_tensor

    def encode_smiles(self, smiles: str) -> torch.Tensor:
        """
        Encodes a single glycan SMILES string into its fingerprint representation.

        Args:
            smiles (str): The SMILES string of the glycan.

        Returns:
            torch.Tensor: A 1D tensor embedding of size `embedding_dim`.
        """
        return self._fingerprint_from_smiles(smiles)

    def encode_batch(self, batch_data: List[str]) -> torch.Tensor:
        """
        Encodes a batch of glycan SMILES strings into their fingerprint representations.

        Args:
            batch_data (List[str]): A list of SMILES strings.

        Returns:
            torch.Tensor: A 2D tensor where each row is the fingerprint embedding of a glycan.
        """
        fingerprints = [self._fingerprint_from_smiles(smiles) for smiles in batch_data]
        return torch.stack(fingerprints)

    @property
    def embedding_dim(self) -> int:
        """
        Returns the dimensionality of the fingerprint embedding.

        Returns:
            int: The number of bits in the fingerprint.
        """
        return self._embedding_dim

    def to(self, device):
        """
        Mimics the .to() method for compatibility with PyTorch modules.
        RDKit operations run on the CPU, so this is effectively a no-op.

        Args:
            device: The device to which the encoder is moved.

        Returns:
            self
        """
        # No device-specific operations are needed for RDKit-based computations.
        return self
