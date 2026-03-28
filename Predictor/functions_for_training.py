import numpy as np
import re
import sys
import os
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.feature_selection import VarianceThreshold
from itertools import product
import logging

module_dir = os.path.dirname(os.path.abspath(__file__))

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
kmers_2 = [''.join(aa) for aa in product(amino_acids, repeat=2)]
all_kmers = kmers_2

def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            cleaned_sequence = ''.join(aa for aa in str(record.seq).upper() if aa in amino_acids)
            if cleaned_sequence:
                sequences.append(cleaned_sequence)
    return sequences

def load_motifs(file_path=None):
    """
    Load motifs from a local file. We look inside the Predictor folder first so
    the package works standalone, and fall back to the parent directory for
    backward compatibility.
    """
    search_paths = []
    if file_path:
        search_paths.append(file_path)
    search_paths.append(os.path.join(module_dir, 'protein_motifs.txt'))
    search_paths.append(os.path.join(os.path.dirname(module_dir), 'protein_motifs.txt'))

    for path in search_paths:
        motifs = {}
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '#' in line:
                            pattern, name = line.split('#', 1)
                            motifs[name.strip()] = pattern.strip()
            return motifs

    return {}

motifs = load_motifs()

aa_properties = {
    'A': [1.8, 0.0, 0.0, 0.008, 0.134, -0.475, -0.039, 0.181],
    'R': [-4.5, 3.0, 52.0, 0.171, -0.361, 0.107, -0.258, -0.364],
    'N': [-3.5, 0.2, 3.38, 0.255, 0.038, 0.117, 0.118, -0.055],
    'D': [-3.5, -1.0, 49.7, 0.303, -0.057, -0.014, 0.225, 0.156],
    'C': [2.5, 0.0, 1.48, -0.132, 0.174, 0.070, 0.565, -0.374],
    'Q': [-3.5, 0.2, 3.53, 0.149, -0.184, -0.030, 0.035, 0.112],
    'E': [-3.5, -1.0, 49.9, 0.221, -0.280, -0.315, 0.157, 0.303],
    'G': [-0.4, 0.0, 0.0, 0.218, 0.562, -0.024, 0.018, 0.106],
    'H': [-3.2, 0.1, 51.6, 0.023, -0.177, 0.041, 0.280, -0.021],
    'I': [4.5, 0.0, 0.13, -0.353, 0.071, -0.088, -0.195, -0.107],
    'L': [3.8, 0.0, 0.13, -0.267, 0.018, -0.265, -0.274, 0.206],
    'K': [-3.9, 1.0, 49.5, 0.243, -0.339, -0.044, -0.325, -0.027],
    'M': [1.9, 0.0, 1.43, -0.239, -0.141, -0.155, 0.321, 0.077],
    'F': [2.8, 0.0, 0.35, -0.329, -0.023, 0.072, -0.002, 0.208],
    'P': [-1.6, 0.0, 1.58, 0.173, 0.286, 0.407, -0.215, 0.384],
    'S': [-0.8, 0.0, 1.67, 0.199, 0.238, -0.015, -0.068, -0.196],
    'T': [-0.7, 0.0, 1.66, 0.068, 0.147, -0.015, -0.132, -0.274],
    'W': [-0.9, 0.0, 2.1, -0.296, -0.186, 0.389, 0.083, 0.297],
    'Y': [-1.3, 0.0, 1.61, -0.141, -0.057, 0.425, -0.096, -0.091],
    'V': [4.2, 0.0, 0.13, -0.274, 0.136, -0.187, -0.196, -0.299]
}

hydrophobicity_scale = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
}

def calculate_additional_features(sequence):
    seq_array = np.array(list(sequence))
    length = len(sequence)

    # Aliphatic index
    aliphatic_counts = np.array([
        np.sum(seq_array == 'A'),
        np.sum(seq_array == 'V') * 2.9,
        (np.sum(seq_array == 'I') + np.sum(seq_array == 'L')) * 3.9
    ])
    aliphatic_index = np.sum(aliphatic_counts) / length * 100

    # Sequence entropy
    unique, counts = np.unique(seq_array, return_counts=True)
    freqs = counts / length
    entropy = -np.sum(freqs * np.log2(freqs))

    # Sequence repetitiveness
    from numpy.lib.stride_tricks import sliding_window_view
    if len(sequence) >= 3:
        windows = sliding_window_view(seq_array, 3)
        window_strings = np.array([''.join(w) for w in windows])
        unique_windows, window_counts = np.unique(window_strings, return_counts=True)
        repeats = np.sum(window_counts) / length
    else:
        repeats = 0

    # Hydrophobic moment 
    hydrophobicity_values = np.array([hydrophobicity_scale.get(aa, 0) for aa in sequence])
    angles = np.arange(len(sequence)) * 100 * np.pi / 180
    hm_sin = np.sum(hydrophobicity_values * np.sin(angles))
    hm_cos = np.sum(hydrophobicity_values * np.cos(angles))
    hydrophobic_moment = np.sqrt(hm_sin**2 + hm_cos**2) / length

    # Charge distribution
    charge_dict = {'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1}
    charges = [charge_dict.get(aa, 0) for aa in sequence]
    charge_distribution = np.std(charges)

    # Polar and non-polar ratio
    polar = sum(sequence.count(aa) for aa in 'QNHSTYC')
    non_polar = sum(sequence.count(aa) for aa in 'AVLIMFPWG')
    polar_non_polar_ratio = polar / non_polar if non_polar > 0 else float('inf')

    # Tiny, small, and large residue composition
    tiny = sum(sequence.count(aa) for aa in 'ACGST') / len(sequence)
    small = sum(sequence.count(aa) for aa in 'ACDGNPSTV') / len(sequence)
    large = sum(sequence.count(aa) for aa in 'EFHIKLMQRWY') / len(sequence)

    # Proline and Cysteine content
    proline_content = sequence.count('P') / len(sequence)
    cysteine_content = sequence.count('C') / len(sequence)

    # Residue bigram transition frequencies
    residue_types = {'polar': 'QNHSTYC', 'non_polar': 'AVLIMFPWG', 'acidic': 'DE', 'basic': 'RK'}
    bigram_transitions = {}
    for type1, type2 in product(residue_types.keys(), repeat=2):
        count = sum(1 for i in range(len(sequence)-1) if sequence[i] in residue_types[type1] and sequence[i+1] in residue_types[type2])
        bigram_transitions[f'{type1}_to_{type2}'] = count / (len(sequence) - 1)

    # Simple E-descriptors features
    e1_sum = sum(aa_properties.get(aa, [0]*8)[3] for aa in sequence)
    e2_sum = sum(aa_properties.get(aa, [0]*8)[4] for aa in sequence)
    e3_sum = sum(aa_properties.get(aa, [0]*8)[5] for aa in sequence)
    e4_sum = sum(aa_properties.get(aa, [0]*8)[6] for aa in sequence)
    e5_sum = sum(aa_properties.get(aa, [0]*8)[7] for aa in sequence)

    return {
        'aliphatic_index': aliphatic_index,
        'sequence_entropy': entropy,
        'sequence_repetitiveness': repeats,
        'hydrophobic_moment': hydrophobic_moment,
        'charge_distribution': charge_distribution,
        'polar_non_polar_ratio': polar_non_polar_ratio,
        'tiny_residues': tiny,
        'small_residues': small,
        'large_residues': large,
        'proline_content': proline_content,
        'cysteine_content': cysteine_content,
        **bigram_transitions,
        'e1_sum': e1_sum,
        'e2_sum': e2_sum,
        'e3_sum': e3_sum,
        'e4_sum': e4_sum,
        'e5_sum': e5_sum
    }

def calculate_edescriptor_features(sequence, aa_properties):
    import numpy as np
    from scipy.spatial.distance import pdist
    e_vectors = np.array([aa_properties[aa][3:8] for aa in sequence if aa in aa_properties])

    if len(e_vectors) == 0:
        return {}

    # Limit sequence length for E-descriptor features to avoid memory issues
    # E-descriptor features are computationally expensive for long sequences
    max_seq_length = 5000
    if len(e_vectors) > max_seq_length:
        logging.warning(f"Sequence length {len(e_vectors)} exceeds limit {max_seq_length}, using first {max_seq_length} residues")
        e_vectors = e_vectors[:max_seq_length]

    # Basic features
    avg_features = {f'avg_e{i+1}': val for i, val in enumerate(np.mean(e_vectors, axis=0))}
    sum_features = {f'sum_e{i+1}': val for i, val in enumerate(np.sum(e_vectors, axis=0))}
    std_features = {f'std_e{i+1}': val for i, val in enumerate(np.std(e_vectors, axis=0))}

    # Vector statistics features
    avg_vector = np.mean(e_vectors, axis=0)
    magnitude = np.linalg.norm(avg_vector)

    # Pairwise distances
    distances = pdist(e_vectors)

    # Vector angles
    norms = np.linalg.norm(e_vectors, axis=1)
    normalized_vectors = e_vectors / norms[:, np.newaxis]

    # Pairwise angles
    cosine_distances = pdist(normalized_vectors, metric='cosine')
    angles = np.arccos(np.clip(1 - cosine_distances, -1.0, 1.0))

    # Distribution statistics
    max_distance = np.max(distances) if len(distances) > 0 else 0
    min_distance = np.min(distances) if len(distances) > 0 else 0
    avg_distance = np.mean(distances) if len(distances) > 0 else 0
    std_distance = np.std(distances) if len(distances) > 0 else 0

    # Angle statistics
    avg_angle = np.mean(angles) if len(angles) > 0 else 0
    std_angle = np.std(angles) if len(angles) > 0 else 0

    # Covariance analysis
    covariance_matrix = np.cov(e_vectors.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    explained_variance_ratio = np.abs(sorted_eigenvalues) / np.sum(np.abs(sorted_eigenvalues))

    # Shape descriptors
    sphericity = (np.prod(np.abs(sorted_eigenvalues))**(1/5)) / np.mean(np.abs(sorted_eigenvalues))
    planarity = (sorted_eigenvalues[1] - sorted_eigenvalues[2]) / sorted_eigenvalues[0] if sorted_eigenvalues[0] != 0 else 0
    linearity = (sorted_eigenvalues[0] - sorted_eigenvalues[1]) / sorted_eigenvalues[0] if sorted_eigenvalues[0] != 0 else 0

    # Vector field
    vector_changes = np.diff(e_vectors, axis=0)
    vector_change_norms = np.linalg.norm(vector_changes, axis=1)
    avg_vector_change = np.mean(vector_change_norms)
    max_vector_change = np.max(vector_change_norms)

    # Entropy measures
    vector_entropy = -np.sum(normalized_vectors * np.log2(np.abs(normalized_vectors) + 1e-10)) / len(sequence)

    # Sequence position weighted features
    position_weights = np.linspace(0.5, 1.5, len(sequence))
    weighted_vectors = e_vectors * position_weights[:, np.newaxis]
    weighted_avg = np.mean(weighted_vectors, axis=0)

    # Local structure indicators
    window_size = 3
    from numpy.lib.stride_tricks import sliding_window_view
    if len(e_vectors) >= window_size:
        windows = sliding_window_view(e_vectors, (window_size, e_vectors.shape[1]))
        windows_reshaped = windows.reshape(-1, window_size, e_vectors.shape[1])
        local_complexity = np.array([np.linalg.det(np.cov(window.T)) for window in windows_reshaped])
        avg_local_complexity = np.mean(local_complexity)
    else:
        avg_local_complexity = 0

    # Combine all advance E-descriptors features
    all_features = {
        **avg_features,
        **sum_features,
        **std_features,
        'e_vector_magnitude': magnitude,
        'e_max_distance': max_distance,
        'e_min_distance': min_distance,
        'e_avg_distance': avg_distance,
        'e_std_distance': std_distance,
        'e_avg_angle': avg_angle,
        'e_std_angle': std_angle,
        'e_sphericity': sphericity,
        'e_planarity': planarity,
        'e_linearity': linearity,
        'e_vector_entropy': vector_entropy,
        'e_avg_vector_change': avg_vector_change,
        'e_max_vector_change': max_vector_change,
        'e_local_complexity': avg_local_complexity,
    }

    # Add eigenvalue features
    for i, ev in enumerate(sorted_eigenvalues):
        all_features[f'e_eigenvalue_{i+1}'] = ev
        all_features[f'e_explained_variance_{i+1}'] = explained_variance_ratio[i]

    # Add weighted vector features
    for i, val in enumerate(weighted_avg):
        all_features[f'e_weighted_avg_{i+1}'] = val

    return all_features

def extract_features(sequence):
    try:
        # Ensure sequence is a string
        if not isinstance(sequence, str):
            sequence = str(sequence)
        sequence = ''.join(aa for aa in sequence if aa in amino_acids)
        if not sequence:
            return None, None
        features = {}

        # Basic BioPython features
        try:
            analysis = ProteinAnalysis(sequence)
            features.update({
                'length': len(sequence),
                'weight': analysis.molecular_weight(),
                'pI': analysis.isoelectric_point(),
                'helix': analysis.secondary_structure_fraction()[0],
                'beta': analysis.secondary_structure_fraction()[1],
                'coil': analysis.secondary_structure_fraction()[2],
                'gravy': analysis.gravy(),
                'aromaticity': analysis.aromaticity(),
                'instability_index': analysis.instability_index(),
            })
        except Exception as e:
            logging.warning(f"Error calculating basic features: {str(e)}")
            return None, None

        # Additional features
        try:
            additional = calculate_additional_features(sequence)
            features.update(additional)
        except Exception as e:
            logging.warning(f"Error calculating additional features: {str(e)}")
            return None, None

        # E-descriptor features
        try:
            edesc = calculate_edescriptor_features(sequence, aa_properties)
            features.update(edesc)
        except Exception as e:
            logging.warning(f"Error calculating E-descriptor features: {str(e)}")
            return None, None

        # Motif features
        try:
            for motif_name, pattern in motifs.items():
                features[f"motif_{motif_name}"] = len(re.findall(pattern, sequence))
        except Exception as e:
            logging.warning(f"Error calculating motif features: {str(e)}")
            return None, None

        # Initialize all possible k-mer features to 0
        for k in [2]:
            for kmer in [''.join(p) for p in product(amino_acids, repeat=k)]:
                features[f"kmer_{kmer}"] = 0

        # Calculate k-mer frequencies
        try:
            k = 2
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i:i+k]
                if all(aa in amino_acids for aa in kmer):  
                    features[f"kmer_{kmer}"] = features.get(f"kmer_{kmer}", 0) + 1
        except Exception as e:
            logging.warning(f"Error calculating k-mer features: {str(e)}")
            return None, None

        # Convert to list with consistent ordering
        feature_names = sorted(features.keys())
        feature_values = [features[name] for name in feature_names]

        # Validate feature vector
        if any(not isinstance(x, (int, float)) for x in feature_values):
            logging.warning("Invalid feature values detected")
            return None, None

        return feature_values, feature_names

    except Exception as e:
        logging.error(f"Error in extract_features: {str(e)}")
        return None, None

def sequences_to_vectors(sequences):
    try:
        logging.info(f"Starting feature extraction for {len(sequences)} sequences...")

        # First extract features from all sequences
        results = []
        for idx, seq in enumerate(sequences):
            features, names = extract_features(seq)
            if features is not None and names is not None:
                results.append((features, names))

            if (idx + 1) % 100 == 0:
                logging.info(f"Processed {idx + 1} sequences")

        if not results:
            raise ValueError("No valid feature vectors were extracted")

        # Verify all feature vectors have the same features
        reference_names = results[0][1]
        valid_results = []
        failed_indices = []

        for idx, (features, names) in enumerate(results):
            if names == reference_names:
                valid_results.append(features)
            else:
                failed_indices.append(idx)
                logging.warning(f"Sequence {idx} has inconsistent features")

        if not valid_results:
            raise ValueError("No sequences with consistent features found")

        feature_matrix = np.array(valid_results)
        logging.info(f"Final feature matrix shape: {feature_matrix.shape}")

        return feature_matrix, reference_names, np.array(failed_indices)

    except Exception as e:
        logging.error(f"Error in sequences_to_vectors: {str(e)}")
        raise

def remove_constant_features(X, feature_names):
    try:
        selector = VarianceThreshold(threshold=0)
        X_without_constant = selector.fit_transform(X)

        # Get the mask of selected features
        constant_feature_mask = selector.get_support()

        # Ensure feature_names is a list for proper indexing
        feature_names = list(feature_names)

        # Select only the feature names that correspond to non-constant features
        selected_feature_names = [name for name, keep in zip(feature_names, constant_feature_mask) if keep]

        print(f"Removed {len(feature_names) - len(selected_feature_names)} constant features")

        return X_without_constant, constant_feature_mask, selected_feature_names

    except Exception as e:
        print(f"Error in remove_constant_features: {str(e)}")
        raise

