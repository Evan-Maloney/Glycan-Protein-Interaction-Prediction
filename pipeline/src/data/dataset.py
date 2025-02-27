import torch
from torch.utils.data import Dataset, random_split, Subset
import pandas as pd
from typing import Dict, Tuple, List
from tqdm import tqdm
from ..base.encoders import GlycanEncoder, ProteinEncoder
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

class BindingDataset(Dataset):
    def __init__(self, 
                 data_frame: pd.DataFrame,
                 glycan_encoder: GlycanEncoder,
                 protein_encoder: ProteinEncoder):
        """
        Dataset for glycan-protein binding data with encodings for unique molecules
        
        Args:
            data_frame: DataFrame with required columns
            glycan_encoder: Encoder for glycan SMILES
            protein_encoder: Encoder for protein sequences
        """
        self.df = data_frame
        
        # Encode unique glycans and proteins
        self.glycan_to_index, self.glycan_encodings = self._encode_unique_glycans(glycan_encoder)
        self.protein_to_index, self.protein_encodings = self._encode_unique_proteins(protein_encoder)
    
    def _encode_unique_glycans(self, glycan_encoder: GlycanEncoder) -> Tuple[Dict[str, int], torch.Tensor]:
        """
        Encode each unique glycan in the dataset once
        
        Args:
            glycan_encoder: Encoder for glycan SMILES
            
        Returns:
            Tuple of (mapping from SMILE to index, tensor of glycan encodings)
        """
        print("Encoding unique glycans...")
        
        # Get unique glycan SMILES
        unique_glycans = self.df['Glycan SMILE'].unique()
        glycan_to_index = {smile: i for i, smile in enumerate(unique_glycans)}
        
        # Encode each unique glycan
        glycan_encodings = []
        for smile in tqdm(unique_glycans):
            with torch.no_grad():
                encoding = glycan_encoder.encode_smiles(smile)
                glycan_encodings.append(encoding)
        
        return glycan_to_index, torch.stack(glycan_encodings)
    
    def _encode_unique_proteins(self, protein_encoder: ProteinEncoder) -> Tuple[Dict[str, int], torch.Tensor]:
        """
        Encode each unique protein in the dataset once
        
        Args:
            protein_encoder: Encoder for protein sequences
            
        Returns:
            Tuple of (mapping from sequence to index, tensor of protein encodings)
        """
        print("Encoding unique proteins...")
        
        # Get unique protein sequences
        unique_proteins = self.df['Protein Sequence'].unique()
        protein_to_index = {seq: i for i, seq in enumerate(unique_proteins)}
        
        # Encode unique proteins in batches
        protein_encodings = []
        batch_size = 2  # Adjust based on your memory constraints
        
        for i in tqdm(range(0, len(unique_proteins), batch_size)):
            batch_sequences = unique_proteins[i:i + batch_size]
            with torch.no_grad():
                batch_encodings = protein_encoder.encode_batch(batch_sequences)
                if isinstance(batch_encodings, list):
                    protein_encodings.extend(batch_encodings)
                else:
                    # If encode_batch returns a tensor
                    for j in range(len(batch_sequences)):
                        protein_encodings.append(batch_encodings[j])
        
        return protein_to_index, torch.stack(protein_encodings)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Get encodings and other data for an index"""
        row = self.df.iloc[idx]
        
        # Get the corresponding encoding using the mapping
        glycan_idx = self.glycan_to_index[row['Glycan SMILE']]
        protein_idx = self.protein_to_index[row['Protein Sequence']]
        
        return {
            'glycan_encoding': self.glycan_encodings[glycan_idx],
            'protein_encoding': self.protein_encodings[protein_idx],
            'concentration': torch.tensor([row['concentration']], dtype=torch.float32),
            'target': torch.tensor([row['fraction_bound']], dtype=torch.float32)
        }


def cluster_glycans(glycans, radius, fp_size, n_clusters):

    def get_morgan_count_fingerprint(smiles, radius, fp_size):
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {f"mf_{i}": 0 for i in range(fp_size)} 


        #The useChirality parameter in Morgan fingerprints determines whether chirality is considered when encoding a molecule.
        #includeChirality=True = Differentiates between enantiomers (model will treat mirror-image molecules as different)
        #includeChirality=False = Ignores chirality (model will treat mirror-image molecules as the same)
        kid_named_morgan_finger = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=fp_size, includeChirality=True)

        cfp = kid_named_morgan_finger.GetCountFingerprint(mol)  
        bit_counts = cfp.GetNonzeroElements()  

        # Convert to a full fp_size-length feature vector
        fingerprint_vector = {f"mf_{i}": bit_counts.get(i, 0) for i in range(fp_size)}
        return fingerprint_vector

    fingerprint_df = glycans['SMILES'].apply(lambda x: get_morgan_count_fingerprint(x, radius, fp_size)).apply(pd.Series)
    
    glycans = pd.concat([glycans, fingerprint_df], axis=1)
    
    # matrix version of fingerprint features. Each row is a glycan, each column is a fingerprint component shape: (611, 2048)
    finger_counts_matrix = fingerprint_df.values
    # pdist calculates the euclidean distance between the combination of each glycan with every other glycan. Then squareform() turns this into a matrix representation where each row is a glycan and each column is the same list of glycans so we can have a comparison matrix. Shape: (611, 611)
    dist_matrix = squareform(pdist(finger_counts_matrix, metric="euclidean"))
    

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(dist_matrix)
    
    glycans['cluster_label'] = labels
    
    return glycans


def stratified_train_test_split(fractions_df, glycans_df, proteins_df, test_size, random_state):
    """
    Create a stratified train-test split where:
    1. Test set has unique GlycanIDs and ProteinGroups not seen in training
    2. Distribution of cluster_labels for both glycans and proteins is maintained
    
    Parameters:
    -----------
    fractions_df : pandas.DataFrame
        DataFrame containing ['ObjId', 'ProteinGroup', 'Concentration', 'GlycanID', 'f']
    glycans_df : pandas.DataFrame
        DataFrame containing ['Name', 'cluster_label'] where Name maps to GlycanID
    proteins_df : pandas.DataFrame
        DataFrame containing ['ProteinGroup', 'cluster_label']
    test_size : float, default=0.1
        Proportion of data to include in the test set
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    train_indices : numpy.ndarray
        Indices of fractions_df that belong to the training set
    test_indices : numpy.ndarray
        Indices of fractions_df that belong to the test set
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Merge cluster labels from glycans and proteins into fractions
    fractions_with_clusters = fractions_df.copy()
    
    # Map glycan cluster labels
    glycan_cluster_map = dict(zip(glycans_df['Name'], glycans_df['cluster_label']))
    fractions_with_clusters['glycan_cluster'] = fractions_with_clusters['GlycanID'].map(glycan_cluster_map)
    
    # Map protein cluster labels
    protein_cluster_map = dict(zip(proteins_df['ProteinGroup'], proteins_df['cluster_label']))
    fractions_with_clusters['protein_cluster'] = fractions_with_clusters['ProteinGroup'].map(protein_cluster_map)
    
    # Get unique glycans and proteins with their cluster labels
    unique_glycans = glycans_df[['Name', 'cluster_label']].drop_duplicates()
    unique_proteins = proteins_df[['ProteinGroup', 'cluster_label']].drop_duplicates()
    
    # Calculate target counts for each cluster in test set
    glycan_cluster_counts = unique_glycans['cluster_label'].value_counts().to_dict()
    protein_cluster_counts = unique_proteins['cluster_label'].value_counts().to_dict()
    
    glycan_test_counts = {cluster: max(1, int(np.ceil(count * test_size))) 
                         for cluster, count in glycan_cluster_counts.items()}
    protein_test_counts = {cluster: max(1, int(np.ceil(count * test_size))) 
                          for cluster, count in protein_cluster_counts.items()}
    
    # Select glycans and proteins for test set while respecting cluster distributions
    test_glycans = []
    for cluster, target_count in glycan_test_counts.items():
        cluster_glycans = unique_glycans[unique_glycans['cluster_label'] == cluster]['Name'].tolist()
        selected = np.random.choice(cluster_glycans, size=min(target_count, len(cluster_glycans)), replace=False)
        test_glycans.extend(selected)
    
    test_proteins = []
    for cluster, target_count in protein_test_counts.items():
        cluster_proteins = unique_proteins[unique_proteins['cluster_label'] == cluster]['ProteinGroup'].tolist()
        selected = np.random.choice(cluster_proteins, size=min(target_count, len(cluster_proteins)), replace=False)
        test_proteins.extend(selected)
    
    # Create train and test masks
    is_test = ((fractions_with_clusters['GlycanID'].isin(test_glycans)) | 
               (fractions_with_clusters['ProteinGroup'].isin(test_proteins)))
    
    test_indices = fractions_with_clusters[is_test].index
    train_indices = fractions_with_clusters[~is_test].index
    
    #train_data = fractions_df.loc[train_indices]
    #test_data = fractions_df.loc[test_indices]
    
    return train_indices, test_indices




def prepare_datasets(
    predict_df: pd.DataFrame,
    glycans_df: pd.DataFrame,
    proteins_df: pd.DataFrame,
    val_split: float,
    glycan_encoder: GlycanEncoder,
    protein_encoder: ProteinEncoder
) -> Tuple[Dataset, Dataset]:
    """
    Prepare train and validation datasets
    
    Args:
        df: Full dataset DataFrame
        val_split: Fraction of data to use for validation
        glycan_encoder: Encoder for glycans
        protein_encoder: Encoder for proteins
    
    Returns:
        Tuple of train and validation datasets
    """
    #full_dataset = BindingDataset(df, glycan_encoder, protein_encoder)
    
    # will move this to config
    random_state = 42
    radius = 3
    fp_size = 1024
    n_clusters = 3
    glycans_df = cluster_glycans(glycans_df, radius, fp_size, n_clusters)
    
    #temp for now
    proteins_df['cluster_label'] = 0
    
    
    train_indices, val_indices = stratified_train_test_split(predict_df, glycans_df, proteins_df, val_split, random_state)
    
    
    
    # Create PyTorch Subset objects
    train_dataset = Subset(predict_df, train_indices)
    val_dataset = Subset(predict_df, val_indices)
    
    
    
    #val_size = int(len(full_dataset) * val_split)
    #train_size = len(full_dataset) - val_size
    
    #train_dataset, val_dataset = random_split(
    #    full_dataset,
    #    [train_size, val_size]
    #)
    
    return train_dataset, val_dataset