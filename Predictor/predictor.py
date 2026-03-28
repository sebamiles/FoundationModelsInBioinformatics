#!/usr/bin/env python3
"""
IApred-PFN Predictor

A command-line tool for predicting protein antigenicity using the TabPFN model.
Supports single sequences or FASTA files as input, with multiple output formats.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from joblib import load
import logging

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(__file__))

from functions_for_training import sequences_to_vectors, extract_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IApredPredictor:
    """Predictor class for IApred-PFN antigenicity predictions"""

    def __init__(self, model_dir='models'):
        """Initialize the predictor with model components"""
        self.model_dir = model_dir
        self.model = None
        self.variance_selector = None
        self.feature_selector = None
        self.scaler = None
        self.all_feature_names = None
        self.is_svm = False

        self._load_model_components()

    def _load_model_components(self):
        """Load all model components from the model directory"""
        try:
            # Check if this is the IApred SVM model
            model_files = list(Path(self.model_dir).glob('IApred_*.joblib'))
            for f in model_files:
                if 'SVM' in f.name:
                    self.is_svm = True
                    break

            # Load model components
            if self.is_svm:
                self.model = load(os.path.join(self.model_dir, 'IApred_SVM.joblib'))
                logger.info("Loaded IApred SVM model")
            else:
                self.model = load(os.path.join(self.model_dir, 'IApred_TabPFN.joblib'))
                logger.info("Loaded TabPFN model")

            self.variance_selector = load(os.path.join(self.model_dir, 'IApred_variance_selector.joblib'))
            self.all_feature_names = load(os.path.join(self.model_dir, 'IApred_all_feature_names.joblib'))

            # Try to load feature selector (may not exist for all_features model)
            selector_path = os.path.join(self.model_dir, 'IApred_feature_selector.joblib')
            if os.path.exists(selector_path):
                self.feature_selector = load(selector_path)
                logger.info("Loaded feature selector")
            else:
                logger.info("No feature selector found (using all features)")

            # Try to load scaler (for IApred SVM and other models)
            scaler_path = os.path.join(self.model_dir, 'IApred_scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = load(scaler_path)
                logger.info("Loaded scaler")
            else:
                logger.info("No scaler found")

            logger.info("Model components loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}")
            raise

    def _preprocess_sequences(self, sequences):
        """Extract and preprocess features from sequences"""
        logger.info(f"Extracting features from {len(sequences)} sequences...")

        # Extract features using the same logic as training
        # We need to ensure we get features in the same order as training
        X_new_list = []
        valid_sequences = []

        for seq in sequences:
            features, names = extract_features(seq)
            if features is not None and names is not None:
                # Create aligned feature vector matching training feature order
                feature_map = {name: i for i, name in enumerate(names)}
                aligned_features = np.zeros(len(self.all_feature_names))

                for i, expected_feature in enumerate(self.all_feature_names):
                    if expected_feature in feature_map:
                        aligned_features[i] = features[feature_map[expected_feature]]
                    # Missing features remain 0

                X_new_list.append(aligned_features)
                valid_sequences.append(seq)

        if not X_new_list:
            logger.error("No valid sequences after feature extraction")
            return np.array([]), []

        X_new_aligned = np.array(X_new_list)
        logger.info(f"Feature matrix shape: {X_new_aligned.shape}")

        # Handle NaN and infinite values
        X_new_aligned = np.nan_to_num(X_new_aligned, nan=0.0,
                                     posinf=np.finfo(np.float64).max,
                                     neginf=np.finfo(np.float64).min)
        X_new_aligned = np.clip(X_new_aligned, -1e6, 1e6)

        # Apply variance selector
        if self.variance_selector is not None:
            X_new_filtered = self.variance_selector.transform(X_new_aligned)
        else:
            X_new_filtered = X_new_aligned

        # Apply feature selector if it exists
        if self.feature_selector is not None:
            X_new_final = self.feature_selector.transform(X_new_filtered)
        else:
            X_new_final = X_new_filtered

        logger.info(f"Final feature matrix shape: {X_new_final.shape}")
        return X_new_final, valid_sequences

    def predict(self, sequences):
        """Make predictions for a list of sequences"""
        if not sequences:
            return []

        # Preprocess sequences
        X_processed, valid_sequences = self._preprocess_sequences(sequences)

        if len(valid_sequences) == 0:
            logger.error("No valid sequences after preprocessing")
            return []

        # Make predictions
        logger.info("Making predictions...")

        if self.is_svm:
            # IApred SVM: use decision_function (returns raw scores)
            # Positive values = antigen, negative = non-antigen (threshold = 0.0)
            # Invert scores because the model may have inverted labels
            raw_scores = -self.model.decision_function(X_processed)
            # Convert to probability-like scores for display (sigmoid transformation)
            scores = 1 / (1 + np.exp(-raw_scores))

            # Determine categories based on threshold 0.0 (raw scores)
            categories = ['antigen' if score > 0 else 'non-antigen' for score in raw_scores]
        else:
            # TabPFN: class 0 = 'antigen', class 1 = 'non-antigen'
            # Higher probability of class 0 = more antigenic
            scores = self.model.predict_proba(X_processed)[:, 0]  # Probability of antigen class

            # Determine categories based on threshold
            threshold = 0.5
            categories = ['antigen' if score >= threshold else 'non-antigen' for score in scores]

        # Create results
        results = []
        for seq, score, category in zip(valid_sequences, scores, categories):
            results.append({
                'sequence': seq,
                'score': score,
                'category': category
            })

        return results

    def predict_from_fasta(self, fasta_file):
        """Load sequences from FASTA file and predict"""
        sequences = []
        headers = []

        logger.info(f"Reading sequences from {fasta_file}...")
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                # Clean sequence (keep only standard amino acids)
                cleaned_seq = ''.join(aa for aa in str(record.seq).upper()
                                    if aa in 'ACDEFGHIKLMNPQRSTVWY')
                if cleaned_seq:
                    sequences.append(cleaned_seq)
                    headers.append(record.id)
        except Exception as e:
            logger.error(f"Error reading FASTA file: {str(e)}")
            return []

        if not sequences:
            logger.error("No valid sequences found in FASTA file")
            return []

        # Make predictions
        predictions = self.predict(sequences)

        # Add headers to results
        for i, pred in enumerate(predictions):
            if i < len(headers):
                pred['header'] = headers[i]

        return predictions

def print_results(results, show_scores=True):
    """Print results to terminal"""
    if not results:
        print("No results to display")
        return

    print("\nPrediction Results:")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        header = result.get('header', f'Sequence_{i}')
        sequence = result['sequence']
        score = result['score']
        category = result['category']

        print(f"Sequence {i}: {header}")
        if len(sequence) > 50:
            print(f"  Sequence: {sequence[:25]}...{sequence[-25:]} (length: {len(sequence)})")
        else:
            print(f"  Sequence: {sequence}")
        if show_scores:
            print(f"  Score: {score:.4f}")
        print(f"  Category: {category}")
        print("-" * 40)

def save_csv(results, output_file):
    """Save results to CSV file"""
    if not results:
        logger.warning("No results to save")
        return

    # Prepare data for CSV
    csv_data = []
    for result in results:
        row = {
            'Header': result.get('header', ''),
            'Sequence': result['sequence'],
            'Score': result['score'],
            'Category': result['category']
        }
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

def save_fasta(results, output_file):
    """Save results to FASTA file with scores in headers"""
    if not results:
        logger.warning("No results to save")
        return

    with open(output_file, 'w') as f:
        for i, result in enumerate(results, 1):
            header = result.get('header', f'Sequence_{i}')
            sequence = result['sequence']
            score = result['score']
            category = result['category']

            # Create new header with score and category
            new_header = f"{header}|score={score:.4f}|category={category}"

            f.write(f">{new_header}\n")
            f.write(f"{sequence}\n")

    logger.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='IApred-PFN: Protein Antigenicity Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single sequence
  python predictor.py -s "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL" --csv results.csv

  # Predict from FASTA file
  python predictor.py -f input.fasta --csv results.csv --fasta output.fasta

  # Predict and show only terminal output
  python predictor.py -f input.fasta
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-s', '--sequence', type=str,
                           help='Single amino acid sequence to predict')
    input_group.add_argument('-f', '--fasta', type=str,
                           help='Path to FASTA file containing sequences')

    # Output options
    parser.add_argument('--csv', type=str,
                       help='Save results to CSV file (columns: Header, Sequence, Score, Category)')
    parser.add_argument('--fasta-out', type=str,
                       help='Save results to FASTA file (with scores and categories in headers)')
    parser.add_argument('--no-scores', action='store_true',
                       help='Hide scores in terminal output')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing model files (default: models)')

    args = parser.parse_args()

    # Initialize predictor
    try:
        predictor = IApredPredictor(model_dir=args.model_dir)
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        sys.exit(1)

    # Get predictions
    if args.sequence:
        # Single sequence prediction
        sequences = [args.sequence]
        results = predictor.predict(sequences)
        if results:
            results[0]['header'] = 'Input_Sequence'
    elif args.fasta:
        # FASTA file prediction
        if not os.path.exists(args.fasta):
            logger.error(f"FASTA file not found: {args.fasta}")
            sys.exit(1)
        results = predictor.predict_from_fasta(args.fasta)
    else:
        logger.error("No input provided")
        sys.exit(1)

    if not results:
        logger.error("No predictions generated")
        sys.exit(1)

    # Display results
    print_results(results, show_scores=not args.no_scores)

    # Save to CSV if requested
    if args.csv:
        save_csv(results, args.csv)

    # Save to FASTA if requested
    if args.fasta_out:
        save_fasta(results, args.fasta_out)

    logger.info("Prediction completed successfully")

if __name__ == "__main__":
    main()
