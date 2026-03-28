import os
import numpy as np
from Bio import SeqIO

# Keep imports local so the Predictor folder can run standalone
from functions_for_training import sequences_to_vectors, remove_constant_features

def read_fasta(file_path):
    sequences = []
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    with open(file_path, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            cleaned_sequence = ''.join(aa for aa in str(record.seq).upper() if aa in amino_acids)
            if cleaned_sequence:
                sequences.append(cleaned_sequence)
    return sequences

def get_data_paths():
    """Get paths to antigens and non-antigens directories relative to project root"""
    # Get project root (parent of Predictor folder)
    predictor_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(predictor_dir)

    # Check if antigens/non-antigens are in data folder first
    antigens_dir = os.path.join(project_root, 'data', 'antigens')
    non_antigens_dir = os.path.join(project_root, 'data', 'non-antigens')

    # Fallback to project root if not found in data folder
    if not os.path.exists(antigens_dir):
        antigens_dir = os.path.join(project_root, 'antigens')
    if not os.path.exists(non_antigens_dir):
        non_antigens_dir = os.path.join(project_root, 'non-antigens')

    # Fallback to Predictor folder if not found in project root
    if not os.path.exists(antigens_dir):
        antigens_dir = os.path.join(predictor_dir, 'antigens')
    if not os.path.exists(non_antigens_dir):
        non_antigens_dir = os.path.join(predictor_dir, 'non-antigens')

    return antigens_dir, non_antigens_dir

def load_training_data():
    """Load all training sequences from antigens and non-antigens directories"""
    antigens_dir, non_antigens_dir = get_data_paths()
    
    antigen_files = [os.path.join(antigens_dir, f) for f in os.listdir(antigens_dir) if f.endswith('.fasta')]
    non_antigen_files = [os.path.join(non_antigens_dir, f) for f in os.listdir(non_antigens_dir) if f.endswith('.fasta')]
    
    print("Reading antigen files...")
    antigens = []
    for file_name in antigen_files:
        try:
            sequences = read_fasta(file_name)
            antigens.extend(sequences)
            print(f"  Loaded {len(sequences)} sequences from {os.path.basename(file_name)}")
        except Exception as e:
            print(f"Warning: Could not read file {file_name}: {str(e)}")
    
    print(f"\nReading non-antigen files...")
    non_antigens = []
    for file_name in non_antigen_files:
        try:
            sequences = read_fasta(file_name)
            non_antigens.extend(sequences)
            print(f"  Loaded {len(sequences)} sequences from {os.path.basename(file_name)}")
        except Exception as e:
            print(f"Warning: Could not read file {file_name}: {str(e)}")
    
    print(f"\nTotal sequences: {len(antigens)} antigens, {len(non_antigens)} non-antigens")
    
    all_sequences = antigens + non_antigens
    labels = np.array(['antigen'] * len(antigens) + ['non-antigen'] * len(non_antigens))
    
    return all_sequences, labels, antigen_files, non_antigen_files

def load_and_extract_features():
    """Load sequences and extract features"""
    all_sequences, labels, antigen_files, non_antigen_files = load_training_data()
    
    print("\nExtracting features...")
    X, feature_names, failed_indices = sequences_to_vectors(all_sequences)
    
    if len(failed_indices) > 0:
        failed_indices = failed_indices.astype(int)
        labels = np.delete(labels, failed_indices)
        print(f"Removed {len(failed_indices)} sequences with failed feature extraction")
    
    print("Filtering constant features...")
    X_filtered, feature_mask, feature_names_filtered = remove_constant_features(X, feature_names)
    
    return X_filtered, labels, feature_names_filtered, antigen_files, non_antigen_files

def _find_eval_file(base_dir, stem):
    candidates = [
        os.path.join(base_dir, f"{stem}.csv"),
        os.path.join(base_dir, f"{stem}.ods"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def read_external_evaluation_data():
    """Load external evaluation CSV/ODS files bundled with the Predictor folder."""
    import pandas as pd

    local_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    fallback_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    antigens_path = (_find_eval_file(local_data_dir, 'External_evaluation_antigens') or
                     _find_eval_file(fallback_dir, 'External_evaluation_antigens'))
    non_antigens_path = (_find_eval_file(local_data_dir, 'External_evaluation_non-antigens') or
                         _find_eval_file(fallback_dir, 'External_evaluation_non-antigens'))

    def _load(path):
        if path is None:
            return None
        if path.lower().endswith('.csv'):
            return pd.read_csv(path)
        return pd.read_excel(path, engine='odf')

    antigens_df = _load(antigens_path)
    non_antigens_df = _load(non_antigens_path)

    return antigens_df, non_antigens_df

