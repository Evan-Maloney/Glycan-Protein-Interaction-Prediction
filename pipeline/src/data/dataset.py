# Claude 3.7 used to generate functions

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
from Bio.SeqUtils.ProtParam import ProteinAnalysis
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

def cluster_proteins(proteins, n_clusters):
    
    
    def compute_protein_features(seq):

        # Add reasoning for feature vectors
        
        # Protein Analysis is a Tool from Biopython
        analysis = ProteinAnalysis(seq)
        features = {}
        
        # The following are Basic Features
        features['length'] = len(seq)
        features['mw'] = analysis.molecular_weight()
        features['instability_index'] = analysis.instability_index()

        features['net_charge_pH7'] = analysis.charge_at_pH(7.0)

        aa_percent = analysis.get_amino_acids_percent()

        # Prompted ChatGPT to ask how to parse a
        # N, Q, S, T: Polar Amino Acids, often involved in hydrogen bonding with glycans
        # K, R: Basic Amino Acids, can form hydrogen bonds and electrostatic bonds
        # D, E: Acidic Amino Acids, can interact with positively charged groups of glycans
        for aa in ['N', 'Q', 'S', 'T', 'K', 'R', 'D', 'E']:
            features[f'frac_{aa}'] = aa_percent.get(aa, 0.0)

    
    # F, Y, W are aromatic amino acids which bind with glycans
        for aa in ['F', 'Y', 'W']:
            features[f'frac_{aa}'] = aa_percent.get(aa, 0.0)
            features['aromatic_binding_score'] = (
            aa_percent.get('F', 0.0) +
            aa_percent.get('Y', 0.0) +
            aa_percent.get('W', 0.0)
        )

        features['aromaticity'] = analysis.aromaticity()

        features['hydrophobicity'] = analysis.gravy()

        return features

    feature_dicts = proteins['Amino Acid Sequence'].apply(compute_protein_features)
    features_df = pd.DataFrame(list(feature_dicts))

    proteins = pd.concat([proteins, features_df], axis=1)
    
    # Select the feature columns (all columns from the feature extraction)
    feature_columns = features_df.columns.tolist()
    feature_data = proteins[feature_columns].values

    # apply k means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    proteins['cluster_label'] = kmeans.fit_predict(feature_data)
    
    return proteins


def stratified_train_test_split(fractions_df, glycans_df, proteins_df, test_size, random_state, mode='AND'):
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
        
        
    if mode == 'AND':
        
        is_test = ((fractions_df['GlycanID'].isin(test_glycans)) & 
                (fractions_df['ProteinGroup'].isin(test_proteins)))

        is_train = ((~fractions_df['GlycanID'].isin(test_glycans)) & 
                        (~fractions_df['ProteinGroup'].isin(test_proteins)))
                
        test_indices = fractions_df[is_test].index

        train_indices = fractions_df[is_train].index
        
        print(f'-------------Test size (% of glycans and proteins as combinations in test set): {test_size*100}% -------------')

        print(f'train size: {len(train_indices)}, test size: {len(test_indices)}, total: {len(fractions_df)}')
                
        print(f'train size: {round((len(train_indices)/len(fractions_df))*100, 2)}%, test size: {round((len(test_indices)/len(fractions_df))*100, 2)}%')
        
        print(f'test size % in terms of test/(training+test) size: {round((len(test_indices)/(len(train_indices)+len(test_indices)))*100, 2)}%')
        
        print(f'Total % of dataset used: {round(((len(train_indices)+len(test_indices))/len(fractions_df))*100, 2)}%\n')
    
    else:
    
        # Create train and test masks
        is_test = ((fractions_with_clusters['GlycanID'].isin(test_glycans)) | 
                (fractions_with_clusters['ProteinGroup'].isin(test_proteins)))
        
        test_indices = fractions_with_clusters[is_test].index
        train_indices = fractions_with_clusters[~is_test].index
    
    
    return train_indices, test_indices


def stratified_kfold_split(fractions_df, glycans_df, proteins_df, n_splits, random_state, mode='AND'):
    """
    Create a stratified k-fold split where each fold:
    1. Contains unique GlycanIDs and ProteinGroups not seen in training
    2. Maintains the distribution of cluster_labels for both glycans and proteins
    
    Parameters:
    -----------
    fractions_df : pandas.DataFrame
        DataFrame containing ['ObjId', 'ProteinGroup', 'Concentration', 'GlycanID', 'f']
    glycans_df : pandas.DataFrame
        DataFrame containing ['Name', 'cluster_label'] where Name maps to GlycanID
    proteins_df : pandas.DataFrame
        DataFrame containing ['ProteinGroup', 'cluster_label']
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    fold_indices : list of tuples
        List of (train_indices, test_indices) pairs for each fold
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Initialize a list to store fold indices
    fold_indices = []
    
    # Create folds for glycans - using glycans_df directly as it contains unique glycan IDs
    # glycan_folds = {0: [[0:20], [20:40], ...], 1: [[0:10], [10:20], ...], 2: [...]}
    glycan_folds = {}
    for cluster in glycans_df['cluster_label'].unique():
        cluster_glycans = glycans_df[glycans_df['cluster_label'] == cluster]['Name'].tolist()
        np.random.shuffle(cluster_glycans)
        
        # Create approximately equal sized groups
        glycan_folds[cluster] = []
        for i in range(n_splits):
            start_idx = int(i * len(cluster_glycans) / n_splits)
            end_idx = int((i + 1) * len(cluster_glycans) / n_splits)
            glycan_folds[cluster].append(cluster_glycans[start_idx:end_idx])
    
    # Create folds for proteins - using proteins_df directly as it contains unique protein IDs
    protein_folds = {}
    for cluster in proteins_df['cluster_label'].unique():
        cluster_proteins = proteins_df[proteins_df['cluster_label'] == cluster]['ProteinGroup'].tolist()
        np.random.shuffle(cluster_proteins)
        
        # Create approximately equal sized groups
        protein_folds[cluster] = []
        for i in range(n_splits):
            start_idx = int(i * len(cluster_proteins) / n_splits)
            end_idx = int((i + 1) * len(cluster_proteins) / n_splits)
            protein_folds[cluster].append(cluster_proteins[start_idx:end_idx])
    
    # for each fold: 0, 1, 2, ... k-1 (k iterations)
    for fold_idx in range(n_splits):
        # Collect test glycans and proteins for this fold
        # test_glycans = [cluster_0_fold_fold_idx + cluster_1_fold_fold_idx + cluster_2_fold_fold_idx]
        test_glycans = []
        for cluster, fold_lists in glycan_folds.items():
            test_glycans.extend(fold_lists[fold_idx])
        
        test_proteins = []
        for cluster, fold_lists in protein_folds.items():
            test_proteins.extend(fold_lists[fold_idx])
            
        
        if mode=='AND':
            is_test = ((fractions_df['GlycanID'].isin(test_glycans)) & 
                   (fractions_df['ProteinGroup'].isin(test_proteins)))
        
            is_train = ((~fractions_df['GlycanID'].isin(test_glycans)) & 
                    (~fractions_df['ProteinGroup'].isin(test_proteins)))
            
            train_indices = fractions_df[is_train].index
        
        else:
        
            # if one of the test_glycans OR one of the test_proteins is in this sample then put it in test, otherwise put it in train
            # becuase of this functionality we need a larger k_fold (something like 8) to get a test_size of around 20% as the OR operation grabs a lot of samples if the test_glycans and test_proteins is high
            # ex: k_fold=2: test_glycans=[50% of our glycans], test_proteins=[50% of our proteins] --> 50% of glycans OR 50% of proteins ~= 80% samples. (This creates a test set of size 80%)
            is_test = ((fractions_df['GlycanID'].isin(test_glycans)) | 
                    (fractions_df['ProteinGroup'].isin(test_proteins)))
            
            train_indices = fractions_df[~is_test].index
            
        
        test_indices = fractions_df[is_test].index
        fold_indices.append((train_indices, test_indices))
    
    
    # fold_indicies at K_fold=2 = [
    #   fold_1: (train_indicies, test_indicies), -> (the_rest, [1/k*total_glycans OR 1/k*total_proteins]) (first half of proteins and glycans in test set)
    #   fold_2: (train_indicies, test_indicies), -> (the_rest, [1/k*total_glycans OR 1/k*total_proteins]) (SECOND half of proteins and glycans in test set)
    #]
    
    return fold_indices


def prepare_kfold_datasets(
    fractions_df: pd.DataFrame,
    glycans_df: pd.DataFrame,
    proteins_df: pd.DataFrame,
    k_folds: float,
    glycan_encoder: GlycanEncoder,
    protein_encoder: ProteinEncoder,
    random_state: int,
    split_mode: str,
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
    
    # for each glycan create a glycan_encoding feature where we use glycan_encoder to encode the SMILES
    # for each protein create a protein_encoding feature where we use protein_encoder to encode the aminoacids
    glycan_encodings = glycan_encoder.encode_batch(glycans_df['SMILES'].tolist())
    protein_encodings = protein_encoder.encode_batch(proteins_df['Amino Acid Sequence'].tolist())
    
    
    # Might move to config but leave for now as our train and test are clusterd and stratified using these parameters
    radius = 3
    fp_size = 1024
    n_clusters = 3
    glycans_df = cluster_glycans(glycans_df, radius, fp_size, n_clusters)
    
    n_protein_clusters = 3
    proteins_df = cluster_proteins(proteins_df, n_protein_clusters)
    
    # need to cluster both glycans and proteins so that we can create a stratified k-fold split for training 
    full_indicies = stratified_kfold_split(fractions_df, glycans_df, proteins_df, k_folds, random_state, split_mode)
    
    
    return full_indicies, glycan_encodings, protein_encodings

def batch_encode(encoder, data_list, device, batch_size):
    """Process data in batches to avoid CUDA memory overflow"""
    all_encodings = []
    total_items = len(data_list)
    
    for i in range(0, total_items, batch_size):
        # Get current batch
        batch = data_list[i:min(i+batch_size, total_items)]
        
        # Encode batch
        batch_encodings = encoder.encode_batch(batch, device)
        all_encodings.append(batch_encodings)
        
        # Print progress
        print(f'Progress: {min(i+batch_size, total_items)}/{total_items}')
        
        # Optional: clear CUDA cache to prevent memory fragmentation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    return torch.cat(all_encodings, dim=0)

def prepare_train_val_datasets(
    fractions_df: pd.DataFrame,
    glycans_df: pd.DataFrame,
    proteins_df: pd.DataFrame,
    glycan_encoder: GlycanEncoder,
    protein_encoder: ProteinEncoder,
    glycan_type: str,
    random_state: int,
    split_mode: str,
    use_kfolds: bool,
    k_folds: float,
    val_split: float,
    device: torch.device
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
    
    # only do batch to not overload RAM of GPU
    if device.type == 'cuda':
        batch_size = 100  # Adjust based on your GPU memory

        # Encode glycans in batches
        glycan_encodings = batch_encode(
            glycan_encoder, 
            glycans_df[glycan_type].tolist(), 
            device, 
            batch_size=batch_size
        )

        # Encode proteins in batches
        protein_encodings = batch_encode(
            protein_encoder, 
            proteins_df['Amino Acid Sequence'].tolist(), 
            device, 
            batch_size=batch_size
        )
    else:
        glycan_encodings = glycan_encoder.encode_batch(glycans_df[glycan_type].tolist(), device)
        protein_encodings = protein_encoder.encode_batch(proteins_df['Amino Acid Sequence'].tolist(), device)
    
    
    # Might move to config but leave for now as our train and test are clusterd and stratified using these parameters
    radius = 3
    fp_size = 1024
    n_clusters = 3
    glycans_df = cluster_glycans(glycans_df, radius, fp_size, n_clusters)
    
    n_protein_clusters = 3
    proteins_df = cluster_proteins(proteins_df, n_protein_clusters)
    
    if use_kfolds:
        # need to cluster both glycans and proteins so that we can create a stratified k-fold split for training 
        full_indicies = stratified_kfold_split(fractions_df, glycans_df, proteins_df, k_folds, random_state, split_mode)
    else:
        train_indicies, test_indicies = stratified_train_test_split(fractions_df, glycans_df, proteins_df, val_split, random_state, split_mode)
        # convert to kfold format so we can use the same code
        full_indicies = [(train_indicies, test_indicies)]
    
    return full_indicies, glycan_encodings, protein_encodings