#!/usr/bin/env python3
"""
Compute Statistical Tests from Existing Evaluation Results

This script reads existing evaluation results and computes statistical tests
(DeLong for AUC, McNemar for accuracy, Wilcoxon for F1) without re-running
the full evaluation pipeline.

Usage:
    python scripts/compute_statistics.py
    python scripts/compute_statistics.py --results-dir results/
"""

import os
import sys
import warnings
import logging
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

# Add Predictor to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Predictor'))

from functions_for_training import extract_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
PROJECT_ROOT = Path(__file__).parent.parent

# Model name mapping for consistent naming
MODEL_NAME_MAPPING = {
    'RandomForest': 'RF (SelectKBest)',
    'XGBoost': 'XGBoost (SelectKBest)',
    'TabPFN': 'TabPFN (SelectKBest)',
    'TabPFN_Default': 'TabPFN (Default)',
    'IApred_SVM': 'IApred(SVM)'
}


def load_existing_results(results_dir):
    """Load existing evaluation results from CSV files

    Args:
        results_dir: Directory containing evaluation results

    Returns:
        Dictionary with model results and evaluation dataframe
    """
    results_dir = Path(results_dir)

    # Load performance metrics
    metrics_file = results_dir / 'tables' / 'table_performance_metrics.csv'
    if not metrics_file.exists():
        logger.error(f"Performance metrics file not found: {metrics_file}")
        return None

    metrics_df = pd.read_csv(metrics_file)
    logger.info(f"Loaded performance metrics from {metrics_file}")

    # Load evaluation data
    evaluation_file = PROJECT_ROOT / 'data' / 'evaluation_antigens' / 'unified_antigens.csv'
    non_evaluation_file = PROJECT_ROOT / 'data' / 'evaluation_non_antigens' / 'unified_non_antigens.csv'

    if not evaluation_file.exists() or not non_evaluation_file.exists():
        logger.error("Evaluation data files not found")
        return None

    # Parse evaluation files
    def parse_evaluation_file(df):
        col = df.columns[0]
        parsed = df[col].str.split(';', expand=True)
        parsed.columns = ['Sequence', 'Class', 'Organism']
        return parsed

    antigens_raw = pd.read_csv(evaluation_file)
    non_antigens_raw = pd.read_csv(non_evaluation_file)

    antigens_df = parse_evaluation_file(antigens_raw)
    non_antigens_df = parse_evaluation_file(non_antigens_raw)

    antigens_df['label'] = 1
    non_antigens_df['label'] = 0

    evaluation_df = pd.concat([antigens_df, non_antigens_df], ignore_index=True)
    logger.info(f"Loaded {len(evaluation_df)} evaluation samples")

    # Load predictions from each model
    models_dir = PROJECT_ROOT / 'models'
    results = {}

    # Map model names from CSV to directory names
    model_dir_mapping = {
        'RF (SelectKBest)': 'RandomForest',
        'XGBoost (SelectKBest)': 'XGBoost',
        'TabPFN (SelectKBest)': 'TabPFN',
        'TabPFN (Default)': 'TabPFN_Default',
        'IApred(SVM)': 'IApred_SVM'
    }

    for model_name_csv in metrics_df['Model'].tolist():
        model_dir_name = model_dir_mapping.get(model_name_csv, model_name_csv)
        model_dir = models_dir / model_dir_name
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            continue

        # Find the model file
        model_files = list(model_dir.glob('IApred_*.joblib'))
        model_file = None
        for f in model_files:
            if 'variance_selector' not in f.name and 'feature_selector' not in f.name and \
               'scaler' not in f.name and 'feature_names' not in f.name and \
               'feature_mask' not in f.name and 'training_metrics' not in f.name:
                model_file = f
                break

        if model_file is None:
            logger.warning(f"No model file found in {model_dir}")
            continue

        # Load model
        from joblib import load
        model = load(model_file)

        # Use the display name from CSV
        model_name = model_name_csv
        
        # Load preprocessing components
        variance_path = model_dir / 'IApred_variance_selector.joblib'
        variance_selector = load(variance_path) if variance_path.exists() else None
        
        scaler_path = model_dir / 'IApred_scaler.joblib'
        scaler = load(scaler_path) if scaler_path.exists() else None
        
        selector_path = model_dir / 'IApred_feature_selector.joblib'
        feature_selector = load(selector_path) if selector_path.exists() else None
        
        feature_names_path = model_dir / 'IApred_all_feature_names.joblib'
        all_feature_names = load(feature_names_path) if feature_names_path.exists() else None
        
        # Load feature names
        if all_feature_names is None:
            feature_names_path = model_dir / 'IApred_feature_names.joblib'
            if feature_names_path.exists():
                all_feature_names = load(feature_names_path)
        
        # Extract features for all sequences
        logger.info(f"Generating predictions for {model_name}...")
        sequences = evaluation_df['Sequence'].tolist()
        
        all_features = []
        for seq in sequences:
            features, names = extract_features(seq)
            if features is not None:
                all_features.append((features, names))
        
        if not all_features:
            logger.error(f"No valid features extracted for {model_name}")
            continue
        
        features_list = [f for f, n in all_features]
        feature_names = all_features[0][1]
        features = np.array(features_list)
        
        # Align features with model's expected feature names
        if all_feature_names is not None:
            feature_idx_map = {name: idx for idx, name in enumerate(feature_names)}
            model_feature_idx_map = {name: idx for idx, name in enumerate(all_feature_names)}
            
            common_features = set(feature_names) & set(all_feature_names)
            
            aligned_features = np.zeros((features.shape[0], len(all_feature_names)))
            for i, feat_name in enumerate(all_feature_names):
                if feat_name in feature_idx_map:
                    aligned_features[:, i] = features[:, feature_idx_map[feat_name]]
            
            features = aligned_features
            feature_names = all_feature_names
        
        # Apply variance filtering
        if variance_selector is not None:
            features = variance_selector.transform(features)
        
        # Apply scaling
        if scaler is not None:
            features = scaler.transform(features)
        
        # Apply feature selection
        if feature_selector is not None:
            features = feature_selector.transform(features)

        # Predict
        if 'SVM' in model_name or model_name == 'IApred_SVM':
            raw_scores = -model.decision_function(features)
            probabilities = 1 / (1 + np.exp(-raw_scores))
        else:
            probabilities = model.predict_proba(features)[:, 1]
        
        # Calculate metrics
        y_true = evaluation_df['label'].values
        y_pred = (probabilities >= 0.5).astype(int)
        
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, accuracy_score,
            precision_score, recall_score, f1_score, matthews_corrcoef,
            confusion_matrix
        )
        
        metrics = {
            'roc_auc': roc_auc_score(y_true, probabilities),
            'pr_auc': average_precision_score(y_true, probabilities),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'specificity': recall_score(y_true, y_pred, pos_label=0),
            'sensitivity': recall_score(y_true, y_pred, pos_label=1),
            'tn': confusion_matrix(y_true, y_pred)[0, 0],
            'fp': confusion_matrix(y_true, y_pred)[0, 1],
            'fn': confusion_matrix(y_true, y_pred)[1, 0],
            'tp': confusion_matrix(y_true, y_pred)[1, 1]
        }

        results[model_name] = {
            'metrics': metrics,
            'probabilities': probabilities,
            'predictions': y_pred
        }
    
    if not results:
        logger.error("No models loaded successfully")
        return None
    
    return {
        'results': results,
        'evaluation_df': evaluation_df
    }


def perform_statistical_tests(results, evaluation_df):
    """Perform statistical tests for model comparison
    
    Args:
        results: Dictionary with evaluation results
        evaluation_df: DataFrame with evaluation data
        
    Returns:
        DataFrame with statistical test results
    """
    logger.info("Performing statistical comparisons...")
    
    models = list(results.keys())
    if len(models) < 2:
        logger.warning("Need at least 2 models for statistical tests")
        return None
    
    y_true = evaluation_df['label'].values
    
    # Collect predictions and probabilities
    predictions = {}
    probabilities = {}
    for model_name in models:
        predictions[model_name] = (results[model_name]['probabilities'] >= 0.5).astype(int)
        probabilities[model_name] = results[model_name]['probabilities']
    
    # Statistical tests for ROC-AUC (DeLong test)
    logger.info("Performing DeLong test for ROC-AUC comparison...")
    delong_results = []
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            auc1 = results[model1]['metrics']['roc_auc']
            auc2 = results[model2]['metrics']['roc_auc']
            
            prob1 = probabilities[model1]
            prob2 = probabilities[model2]
            
            delong_result = compute_delong_auc_significance(y_true, prob1, prob2)
            
            delong_results.append({
                'Model 1': model1,
                'Model 2': model2,
                'AUC Model 1': auc1,
                'AUC Model 2': auc2,
                'AUC Difference': auc1 - auc2,
                'Z-statistic': delong_result['z_stat'],
                'p-value': delong_result['p_value'],
                'Significant (α=0.05)': 'Yes' if delong_result['p_value'] < 0.05 else 'No'
            })
    
    delong_df = pd.DataFrame(delong_results)
    
    # Statistical tests for Accuracy (McNemar's test)
    logger.info("Performing McNemar's test for accuracy comparison...")
    mcnemar_results = []
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            pred1 = predictions[model1]
            pred2 = predictions[model2]
            
            both_correct = np.sum((pred1 == y_true) & (pred2 == y_true))
            only_model1 = np.sum((pred1 == y_true) & (pred2 != y_true))
            only_model2 = np.sum((pred1 != y_true) & (pred2 == y_true))
            both_incorrect = np.sum((pred1 != y_true) & (pred2 != y_true))
            
            mcnemar_result = mcnemar([[both_correct, only_model1], 
                                      [only_model2, both_incorrect]], exact=False)
            
            mcnemar_results.append({
                'Model 1': model1,
                'Model 2': model2,
                'Accuracy Model 1': results[model1]['metrics']['accuracy'],
                'Accuracy Model 2': results[model2]['metrics']['accuracy'],
                'Accuracy Difference': results[model1]['metrics']['accuracy'] - results[model2]['metrics']['accuracy'],
                'Chi-square': mcnemar_result.statistic,
                'p-value': mcnemar_result.pvalue,
                'Significant (α=0.05)': 'Yes' if mcnemar_result.pvalue < 0.05 else 'No'
            })
    
    mcnemar_df = pd.DataFrame(mcnemar_results)
    
    # Statistical tests for F1 score (Wilcoxon signed-rank test)
    logger.info("Performing Wilcoxon signed-rank test for F1 score comparison...")
    wilcoxon_results = []
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            prob1 = probabilities[model1]
            prob2 = probabilities[model2]
            
            pred1 = (prob1 >= 0.5).astype(int)
            pred2 = (prob2 >= 0.5).astype(int)
            
            from sklearn.metrics import f1_score
            f1_1 = np.array([f1_score([y_true[j]], [pred1[j]]) for j in range(len(y_true))])
            f1_2 = np.array([f1_score([y_true[j]], [pred2[j]]) for j in range(len(y_true))])
            
            wilcoxon_result = wilcoxon(f1_1, f1_2, zero_method='pratt')
            
            wilcoxon_results.append({
                'Model 1': model1,
                'Model 2': model2,
                'F1 Model 1': results[model1]['metrics']['f1'],
                'F1 Model 2': results[model2]['metrics']['f1'],
                'F1 Difference': results[model1]['metrics']['f1'] - results[model2]['metrics']['f1'],
                'W-statistic': wilcoxon_result.statistic,
                'p-value': wilcoxon_result.pvalue,
                'Significant (α=0.05)': 'Yes' if wilcoxon_result.pvalue < 0.05 else 'No'
            })
    
    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    
    # Combine results
    stat_tests_df = pd.concat([
        delong_df.rename(columns=lambda x: f'Delong_{x}' if x not in ['Model 1', 'Model 2'] else x),
        mcnemar_df.rename(columns=lambda x: f'McNemar_{x}' if x not in ['Model 1', 'Model 2'] else x),
        wilcoxon_df.rename(columns=lambda x: f'Wilcoxon_{x}' if x not in ['Model 1', 'Model 2'] else x)
    ], axis=1)
    
    stat_tests_df = stat_tests_df.loc[:, ~stat_tests_df.columns.duplicated()]
    
    logger.info("Statistical tests completed")
    return stat_tests_df


def compute_delong_auc_significance(y_true, prob1, prob2):
    """Compute DeLong test for comparing two ROC curves
    
    Args:
        y_true: True binary labels
        prob1: Predicted probabilities for model 1
        prob2: Predicted probabilities for model 2
        
    Returns:
        Dictionary with z-statistic and p-value
    """
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    
    y_true = np.asarray(y_true)
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)
    
    auc1 = roc_auc_score(y_true, prob1)
    auc2 = roc_auc_score(y_true, prob2)
    
    diff = prob1 - prob2
    var_diff = np.var(diff)
    
    se_diff = np.sqrt(var_diff) / np.sqrt(len(y_true))
    if se_diff > 0:
        z_stat = (auc1 - auc2) / se_diff
    else:
        z_stat = 0
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {'z_stat': z_stat, 'p_value': p_value}


def main():
    parser = argparse.ArgumentParser(
        description='Compute statistical tests from existing evaluation results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory containing existing evaluation results (default: results/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for statistical tests (default: same as results-dir)'
    )
    
    args = parser.parse_args()
    
    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Try to find existing results
        results_dir = PROJECT_ROOT / 'results'
        if not (results_dir / 'tables' / 'table_performance_metrics.csv').exists():
            # Try all_4_models subdirectory
            results_dir = PROJECT_ROOT / 'results' / 'all_4_models'
            if not (results_dir / 'tables' / 'table_performance_metrics.csv').exists():
                logger.error("No existing evaluation results found. Please specify --results-dir")
                sys.exit(1)
    
    logger.info(f"Using results directory: {results_dir}")
    
    # Load existing results
    data = load_existing_results(results_dir)
    if data is None:
        sys.exit(1)
    
    results = data['results']
    evaluation_df = data['evaluation_df']
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir
    
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform statistical tests
    stat_tests_df = perform_statistical_tests(results, evaluation_df)
    if stat_tests_df is not None:
        stat_tests_df.to_csv(tables_dir / 'table_statistical_tests.csv', index=False)
        logger.info(f"Statistical tests saved to {tables_dir / 'table_statistical_tests.csv'}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("Statistical Tests Summary")
        print("=" * 80)
        
        # DeLong test summary
        print("\n### DeLong Test for ROC-AUC Comparison ###")
        print(stat_tests_df[['Model 1', 'Model 2', 'AUC Model 1', 'AUC Model 2', 
                             'AUC Difference', 'p-value', 'Significant (α=0.05)']].to_string(index=False))
        
        # McNemar test summary
        print("\n### McNemar's Test for Accuracy Comparison ###")
        print(stat_tests_df[['Model 1', 'Model 2', 'Accuracy Model 1', 'Accuracy Model 2',
                             'Accuracy Difference', 'p-value', 'Significant (α=0.05)']].to_string(index=False))
        
        # Wilcoxon test summary
        print("\n### Wilcoxon Signed-Rank Test for F1 Score Comparison ###")
        print(stat_tests_df[['Model 1', 'Model 2', 'F1 Model 1', 'F1 Model 2',
                             'F1 Difference', 'p-value', 'Significant (α=0.05)']].to_string(index=False))
        
        print("\n" + "=" * 80)
        print("Done!")
        print("=" * 80)
    else:
        logger.error("Failed to compute statistical tests")
        sys.exit(1)


if __name__ == "__main__":
    main()
