import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np

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
    
    # Filter out rows where ProteinGroup is not in proteins_df
    valid_protein_groups = set(proteins_df['ProteinGroup'])
    fractions_df = fractions_df[fractions_df['ProteinGroup'].isin(valid_protein_groups)].copy()

    
    # Calculate target counts for each cluster in test set
    glycan_cluster_counts = glycans_df['cluster_label'].value_counts().to_dict()
    protein_cluster_counts = proteins_df['cluster_label'].value_counts().to_dict()
    
    glycan_test_counts = {cluster: max(1, int(np.ceil(count * test_size))) 
                         for cluster, count in glycan_cluster_counts.items()}
    protein_test_counts = {cluster: max(1, int(np.ceil(count * test_size))) 
                          for cluster, count in protein_cluster_counts.items()}
    
    # Select glycans and proteins for test set while respecting cluster distributions
    test_glycans = []
    for cluster, target_count in glycan_test_counts.items():
        cluster_glycans = glycans_df[glycans_df['cluster_label'] == cluster]['Name'].tolist()
        selected = np.random.choice(cluster_glycans, size=min(target_count, len(cluster_glycans)), replace=False)
        test_glycans.extend(selected)
    
    test_proteins = []
    for cluster, target_count in protein_test_counts.items():
        cluster_proteins = proteins_df[proteins_df['cluster_label'] == cluster]['ProteinGroup'].tolist()
        selected = np.random.choice(cluster_proteins, size=min(target_count, len(cluster_proteins)), replace=False)
        test_proteins.extend(selected)
    
    # Create train and test masks
    is_test = ((fractions_df['GlycanID'].isin(test_glycans)) | 
               (fractions_df['ProteinGroup'].isin(test_proteins)))
    
    test_indices = fractions_df[is_test].index
    train_indices = fractions_df[~is_test].index
    
    print(f"Test data size created: ~{float(np.round(is_test.sum()/len(fractions_df), decimals=4)*100)}%")
    
    train_data = fractions_df.loc[train_indices]
    test_data = fractions_df.loc[test_indices]
    
    return train_data, test_data



def main():
    
    glycans = pd.read_csv('./data/Glycan-Structures-CFG611.txt', sep="\t")
    
    radius = 3
    fp_size = 1024
    n_glycan_clusters = 3

    glycans = cluster_glycans(glycans, radius, fp_size, n_glycan_clusters)
    
    proteins = pd.read_csv('./data/Protein-Sequence-Table.txt', sep='\t')
    
    n_protein_clusters = 3
    proteins = cluster_proteins(proteins, n_protein_clusters)
    
    fractions = pd.read_csv('./data/Fractions-Bound-Table.txt', sep="\t")
    
    # 7% of each cluster and protein cluster class in the test set (at random_state=42 results in a ~15.229% test size)
    test_size = 0.07
    random_state = 42
    
    train_data, test_data = stratified_train_test_split(fractions, glycans, proteins, test_size, random_state)
    
    train_data.to_csv('./data/Train_Fractions.csv', sep='\t')
    
    test_data.to_csv('./data/Test_Fractions.csv', sep='\t')
    
if __name__ == "__main__":
    main()