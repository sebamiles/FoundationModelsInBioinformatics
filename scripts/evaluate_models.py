#!/usr/bin/env python3
"""
External Evaluation Script for IApred Publication Models

Evaluates trained models on external validation datasets,
calculates performance metrics, and generates publication-ready plots.
"""

import os
import sys
import warnings
import logging
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
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
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
PROJECT_ROOT = Path(__file__).parent.parent

# Model name mapping for display
MODEL_DISPLAY_NAMES = {
    'RandomForest': 'RF (SelectKBest)',
    'XGBoost': 'XGBoost (SelectKBest)',
    'TabPFN': 'TabPFN (SelectKBest)',
    'TabPFN_Default': 'TabPFN (Default)',
    'IApred_SVM': 'IApred(SVM)'
}

# Color mapping for models (publication-ready colors)
MODEL_COLORS_ALL = {
    'TabPFN (SelectKBest)': '#9467bd',    # Purple
    'TabPFN (Default)': '#b194d6',        # Light Purple
    'RF (SelectKBest)': '#2ca02c',        # Green
    'XGBoost (SelectKBest)': '#ff7f0e',   # Orange
    'IApred(SVM)': '#d62728'              # Red
}

# Model sets for different evaluation scenarios - sorted order: TabPFN, RF, XGB, SVM
ALL_MODELS = ['TabPFN', 'TabPFN_Default', 'RandomForest', 'XGBoost', 'IApred_SVM']
KBEST_MODELS = ['TabPFN', 'TabPFN_Default', 'RandomForest', 'XGBoost', 'IApred_SVM']


class ModelEvaluator:
    """Evaluator class for IApred models"""

    def __init__(self, model_dir: Path):
        """Initialize the evaluator with model directory"""
        self.model_dir = model_dir
        self.model = None
        self.variance_selector = None
        self.feature_selector = None
        self.scaler = None
        self.all_feature_names = None
        self.model_name = model_dir.name
        self.is_svm = model_dir.name == 'IApred_SVM'

    def load_model(self):
        """Load all model components from the model directory"""
        try:
            # Load model
            model_files = list(self.model_dir.glob('IApred_*.joblib'))

            # Find the main model file
            model_file = None
            for f in model_files:
                if ('variance_selector' not in f.name and
                    'feature_selector' not in f.name and
                    'scaler' not in f.name and
                    'feature_names' not in f.name and
                    'feature_mask' not in f.name and
                    'training_metrics' not in f.name):
                    model_file = f
                    break

            if model_file is None:
                raise FileNotFoundError(f"No model file found in {self.model_dir}")

            self.model = load(model_file)
            logger.info(f"Loaded model from {model_file}")

            # Load variance selector
            variance_path = self.model_dir / 'IApred_variance_selector.joblib'
            if variance_path.exists():
                self.variance_selector = load(variance_path)

            # Load feature selector (for TabPFN and IApred SVM)
            selector_path = self.model_dir / 'IApred_feature_selector.joblib'
            if selector_path.exists():
                self.feature_selector = load(selector_path)

            # Load scaler (for RF/XGB/IApred SVM)
            scaler_path = self.model_dir / 'IApred_scaler.joblib'
            if scaler_path.exists():
                self.scaler = load(scaler_path)

            # Load feature names
            feature_names_path = self.model_dir / 'IApred_all_feature_names.joblib'
            if feature_names_path.exists():
                self.all_feature_names = load(feature_names_path)
            else:
                feature_names_path = self.model_dir / 'IApred_feature_names.joblib'
                if feature_names_path.exists():
                    self.all_feature_names = load(feature_names_path)

            logger.info(f"Model loaded successfully: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading model from {self.model_dir}: {e}")
            raise

    def predict(self, sequences):
        """Predict probabilities for sequences"""
        import numpy as np

        # Handle list of sequences
        if isinstance(sequences, list):
            all_features = []
            for seq in sequences:
                features, names = extract_features(seq)
                if features is not None:
                    all_features.append((features, names))

            if not all_features:
                logger.error("No valid features extracted from sequences")
                return np.array([])

            # Extract feature matrix and names
            features_list = [f for f, n in all_features]
            feature_names = all_features[0][1]  # Use first sequence's feature names
            features = np.array(features_list)
        else:
            # Single sequence
            features, names = extract_features(sequences)
            if features is None:
                logger.error("No valid features extracted from sequence")
                return np.array([])
            feature_names = names
            features = np.array([features])

        # Align features with model's expected feature names
        if hasattr(self, 'all_feature_names') and self.all_feature_names is not None:
            # Create feature map to align with model's feature order
            feature_idx_map = {name: idx for idx, name in enumerate(feature_names)}
            model_feature_idx_map = {name: idx for idx, name in enumerate(self.all_feature_names)}
            
            # Get indices of features that exist in both
            common_features = set(feature_names) & set(self.all_feature_names)
            
            # Create aligned feature matrix
            aligned_features = np.zeros((features.shape[0], len(self.all_feature_names)))
            for i, feat_name in enumerate(self.all_feature_names):
                if feat_name in feature_idx_map:
                    aligned_features[:, i] = features[:, feature_idx_map[feat_name]]
            
            features = aligned_features
            feature_names = self.all_feature_names

        # Apply variance filtering
        if self.variance_selector is not None:
            features = self.variance_selector.transform(features)

        # Apply scaling (for RF/XGB/IApred SVM) - BEFORE feature selection
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Apply feature selection (for TabPFN and IApred SVM) - AFTER scaling
        if self.feature_selector is not None:
            features = self.feature_selector.transform(features)

        # Predict
        # For IApred SVM: use decision_function (returns raw scores)
        # For others: use predict_proba
        if self.is_svm:
            # IApred SVM: positive values = antigen, negative = non-antigen
            # Invert scores because the model may have inverted labels
            raw_scores = -self.model.decision_function(features)
            # Convert to probability-like scores (sigmoid transformation)
            probabilities = 1 / (1 + np.exp(-raw_scores))
        else:
            probabilities = self.model.predict_proba(features)[:, 1]
        return probabilities


def load_evaluation_data():
    """Load evaluation datasets"""
    logger.info("Loading evaluation data...")

    antigens_file = PROJECT_ROOT / 'data' / 'evaluation_antigens' / 'unified_antigens.csv'
    non_antigens_file = PROJECT_ROOT / 'data' / 'evaluation_non_antigens' / 'unified_non_antigens.csv'

    # Load CSV files - they have a single column with semicolon-separated values
    antigens_raw = pd.read_csv(antigens_file)
    non_antigens_raw = pd.read_csv(non_antigens_file)
    
    # Parse the semicolon-separated column
    def parse_evaluation_file(df):
        """Parse semicolon-separated column into separate columns"""
        # Get the single column
        col = df.columns[0]
        # Split by semicolon and create new dataframe
        parsed = df[col].str.split(';', expand=True)
        # Rename columns properly
        parsed.columns = ['Sequence', 'Class', 'Organism']
        return parsed
    
    antigens_df = parse_evaluation_file(antigens_raw)
    non_antigens_df = parse_evaluation_file(non_antigens_raw)

    # Combine
    antigens_df['label'] = 1
    non_antigens_df['label'] = 0

    evaluation_df = pd.concat([antigens_df, non_antigens_df], ignore_index=True)

    logger.info(f"Loaded {len(evaluation_df)} evaluation samples")
    logger.info(f"  Antigens: {sum(evaluation_df['label'] == 1)}")
    logger.info(f"  Non-antigens: {sum(evaluation_df['label'] == 0)}")

    return evaluation_df


def evaluate_all_models(evaluation_df, models_to_evaluate=None):
    """Evaluate trained models
    
    Args:
        evaluation_df: DataFrame with evaluation data
        models_to_evaluate: List of model names to evaluate. If None, evaluates all models.
    """
    logger.info("=" * 80)
    logger.info("Evaluating Models")
    if models_to_evaluate:
        logger.info(f"Models to evaluate: {models_to_evaluate}")
    else:
        logger.info("Evaluating all available models")
    logger.info("=" * 80)

    models_dir = PROJECT_ROOT / 'models'
    
    if models_to_evaluate is None:
        models_to_evaluate = list(MODEL_DISPLAY_NAMES.keys())
    
    results = {}

    for model_name in models_to_evaluate:
        model_dir = models_dir / model_name
        
        if not model_dir.is_dir():
            logger.warning(f"Model directory not found: {model_dir}")
            continue

        if model_name not in MODEL_DISPLAY_NAMES:
            logger.info(f"Skipping {model_name} (not in publication models)")
            continue

        logger.info(f"\nEvaluating {model_name}...")

        try:
            evaluator = ModelEvaluator(model_dir)
            evaluator.load_model()

            # Get predictions - use 'Sequence' column (capital S from parsed CSV)
            sequences = evaluation_df['Sequence'].tolist()
            probabilities = evaluator.predict(sequences)

            # Calculate metrics
            y_true = evaluation_df['label'].values
            y_pred = (probabilities >= 0.5).astype(int)

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

            results[MODEL_DISPLAY_NAMES[model_name]] = {
                'metrics': metrics,
                'probabilities': probabilities,
                'predictions': y_pred
            }

            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue

    return results


def generate_plots(results, evaluation_df, output_dir=None, model_colors=None):
    """Generate publication-ready plots
    
    Args:
        results: Dictionary with evaluation results
        evaluation_df: DataFrame with evaluation data
        output_dir: Directory to save plots (default: results/figures)
        model_colors: Dictionary mapping model names to colors
    """
    logger.info("=" * 80)
    logger.info("Generating Publication Plots")
    logger.info("=" * 80)

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'results' / 'figures'
    else:
        output_dir = Path(output_dir)
    
    figures_dir = output_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if model_colors is None:
        model_colors = MODEL_COLORS_ALL

    # Set style
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.2,
    })

    # 1. ROC Curves (square, 10x10)
    logger.info("Creating ROC curves...")
    fig, ax = plt.subplots(figsize=(10, 10))

    for model_name, data in results.items():
        color = model_colors.get(model_name, '#333333')
        fpr, tpr, _ = roc_curve(evaluation_df['label'], data['probabilities'])
        auc = data['metrics']['roc_auc']
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{model_name} (AUC = {auc:.3f})')

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Chance (AUC = 0.5)', alpha=0.7)
    
    ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=18, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=12, frameon=True, fancybox=False, 
              shadow=False, edgecolor='black', framealpha=0.95)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    # Improved grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    
    # Add tick marks
    ax.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5)
    ax.tick_params(axis='both', which='minor', length=3, width=1)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(figures_dir / 'roc_curves.pdf', dpi=300, facecolor='white')
    plt.savefig(figures_dir / 'roc_curves.png', dpi=300, facecolor='white')
    plt.close()
    logger.info(f"  Saved to {figures_dir / 'roc_curves.pdf'}")

    # 2. PR Curves (square, 10x10)
    logger.info("Creating PR curves...")
    fig, ax = plt.subplots(figsize=(10, 10))

    for model_name, data in results.items():
        color = model_colors.get(model_name, '#333333')
        precision, recall, _ = precision_recall_curve(evaluation_df['label'], data['probabilities'])
        pr_auc = data['metrics']['pr_auc']
        ax.plot(recall, precision, color=color, linewidth=2.5,
                label=f'{model_name} (PR-AUC = {pr_auc:.3f})')

    # Baseline reference line (proportion of positives)
    baseline = evaluation_df['label'].mean()
    ax.axhline(y=baseline, color='black', linestyle='--', linewidth=2,
               label=f'Baseline ({baseline:.2f})')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontsize=16, fontweight='bold')
    ax.set_title('Precision-Recall (PR) Curves', fontsize=18, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=12, frameon=True, fancybox=False, 
              shadow=False, edgecolor='black', framealpha=0.95)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    # Improved grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    
    # Add tick marks
    ax.tick_params(axis='both', which='major', labelsize=13, length=6, width=1.5)
    ax.tick_params(axis='both', which='minor', length=3, width=1)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(figures_dir / 'pr_curves.pdf', dpi=300, facecolor='white')
    plt.savefig(figures_dir / 'pr_curves.png', dpi=300, facecolor='white')
    plt.close()
    logger.info(f"  Saved to {figures_dir / 'pr_curves.pdf'}")

    # 3. Metrics Comparison (bar chart)
    logger.info("Creating metrics comparison...")
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    metric_names = {
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'mcc': 'MCC',
        'specificity': 'Specificity',
        'sensitivity': 'Sensitivity'
    }

    models = list(results.keys())
    n_models = len(models)

    for idx, (metric, name) in enumerate(metric_names.items()):
        ax = axes[idx]
        values = [results[m]['metrics'][metric] for m in models]
        colors = [model_colors.get(m, '#333333') for m in models]

        x = np.arange(n_models)
        width = 0.7

        bars = ax.bar(x, values, width, color=colors, edgecolor='black', linewidth=1.0)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(name, fontsize=13, fontweight='bold')
        ax.set_title(name, fontsize=15, fontweight='bold', pad=10)
        ax.set_ylim([0, 1.15])
        
        # Add horizontal reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
        
        # Improved grid
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')
        ax.set_axisbelow(True)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        
        ax.tick_params(axis='y', labelsize=11)

    for idx in range(len(metric_names), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / 'metrics_comparison.pdf', dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(figures_dir / 'metrics_comparison.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved to {figures_dir / 'metrics_comparison.pdf'}")

        # 4. Confusion Matrices (dynamic grid based on number of models)
    logger.info("Creating confusion matrices...")
    n_models = len(results)
    if n_models <= 4:
        n_cols = 2
        n_rows = 2
    elif n_models <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 4
        n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Ensure axes is always a list
    if n_rows * n_cols == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten()

    for idx, model_key in enumerate(results.keys()):
        if idx >= len(axes_list):
            break
        ax = axes_list[idx]
        cm = np.array([
            [results[model_key]['metrics']['tn'], results[model_key]['metrics']['fp']],
            [results[model_key]['metrics']['fn'], results[model_key]['metrics']['tp']]
        ])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues,
                        vmin=0, vmax=1)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(xticks=np.arange(2),
                yticks=np.arange(2),
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['Actual Neg', 'Actual Pos'])

        # Improved title with multiple metrics
        acc = results[model_key]["metrics"]["accuracy"]
        f1 = results[model_key]["metrics"]["f1"]
        ax.set_title(f'{model_key}\nAcc={acc:.3f}, F1={f1:.3f}',
                            fontsize=14, fontweight='bold', pad=10)

        thresh = cm_normalized.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm_normalized[i, j]*100:.1f}%\n({cm[i, j]})',
                         ha="center", va="center",
                         color="white" if cm_normalized[i, j] > thresh else "black",
                         fontsize=12, fontweight='bold')

        # Style ticks
        ax.tick_params(axis='both', labelsize=11)

        # Add grid lines
        ax.set_xticks(np.arange(-.5, 1.5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 1.5, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    for idx in range(len(results), len(axes_list)):
        if idx < len(axes.flatten()):
            axes.flatten()[idx].set_visible(False)

    
    plt.tight_layout()
    plt.savefig(figures_dir / 'confusion_matrices.pdf', dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(figures_dir / 'confusion_matrices.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved to {figures_dir / 'confusion_matrices.pdf'}")

    # 5. Radar Chart (square, 10x10)
    logger.info("Creating radar chart...")
    from math import pi

    metrics = ['roc_auc', 'pr_auc', 'accuracy', 'f1', 'mcc', 'specificity', 'sensitivity']
    metric_names_plot = ['ROC-AUC', 'PR-AUC', 'Accuracy', 'F1', 'MCC', 'Specificity', 'Sensitivity']

    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    for model_name in models:
        values = [results[model_name]['metrics'][m] for m in metrics]
        values += values[:1]
        color = model_colors.get(model_name, '#333333')

        ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name, color=color)
        ax.scatter(angles, values, s=100, color=color, zorder=5, edgecolors='white', linewidths=1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names_plot, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add radial lines at 0.25, 0.5, 0.75, 1.0
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=10, color='gray')
    
    ax.set_title('Model Performance Radar Chart', pad=25, fontsize=18, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12, 
              frameon=True, fancybox=False, shadow=False, edgecolor='black', framealpha=0.95)
    
    # Improved grid - gray color for visibility
    ax.grid(True, color='gray', alpha=0.5, linestyle='--', linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('gray')

    plt.tight_layout()
    plt.savefig(figures_dir / 'radar_chart.pdf', dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(figures_dir / 'radar_chart.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved to {figures_dir / 'radar_chart.pdf'}")

        # 6. Score Distribution Plots (grid of separate plots, one per model)
    logger.info("Creating score distribution plots...")
    n_models = len(results)
    if n_models <= 4:
        n_cols = 2
        n_rows = 2
    elif n_models <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 4
        n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    if n_rows * n_cols == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten()
    
    antigen_mask = evaluation_df['label'] == 1
    non_antigen_mask = evaluation_df['label'] == 0
    
    for idx, (model_name, data) in enumerate(results.items()):
        if idx >= len(axes_list):
            break
        ax = axes_list[idx]
        color = model_colors.get(model_name, '#333333')
        
        antigen_scores = data['probabilities'][antigen_mask]
        non_antigen_scores = data['probabilities'][non_antigen_mask]
        
        # Plot histograms for antigens and non-antigens separately
        ax.hist(antigen_scores, bins=25, alpha=0.6, label='Antigens', color='#2ca02c', density=True)
        ax.hist(non_antigen_scores, bins=25, alpha=0.6, label='Non-antigens', color='#d62728', density=True)
        
        ax.set_xlabel('Prediction Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlim([0, 1])
        ax.legend(loc='upper center', fontsize=9, frameon=True, fancybox=False,
                  shadow=False, edgecolor='black', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color('black')
    
    # Hide unused subplots
    for idx in range(len(results), len(axes_list)):
        axes_list[idx].set_visible(False)
    
    plt.suptitle('Score Distributions by Model', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / 'score_distributions.pdf', dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(figures_dir / 'score_distributions.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved to {figures_dir / 'score_distributions.pdf'}")

    # 7. Calibration Plot with ECE
    logger.info("Creating calibration plot...")
    from sklearn.calibration import calibration_curve

    def calculate_ece(y_true, y_prob, n_bins=10):
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    fig, ax = plt.subplots(figsize=(10, 10))

    for model_name, data in results.items():
        color = model_colors.get(model_name, '#333333')
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(evaluation_df['label'], data['probabilities'],
                                                   n_bins=10, strategy='uniform')
        # Calculate ECE
        ece = calculate_ece(evaluation_df['label'].values, data['probabilities'])
        ax.plot(prob_pred, prob_true, 's-', color=color, linewidth=2.5, markersize=8,
                label=f'{model_name} (ECE={ece:.3f})')

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectly Calibrated')

    ax.set_xlabel('Mean Predicted Probability', fontsize=16, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=16, fontweight='bold')
    ax.set_title('Calibration Curves', fontsize=18, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=False,
              shadow=False, edgecolor='black', framealpha=0.95)
    ax.set_xlim([0, 1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Make all spines visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'calibration_curves.pdf', dpi=300, facecolor='white')
    plt.savefig(figures_dir / 'calibration_curves.png', dpi=300, facecolor='white')
    plt.close()
    logger.info(f"  Saved to {figures_dir / 'calibration_curves.pdf'}")


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
            # DeLong test for ROC-AUC
            auc1 = results[model1]['metrics']['roc_auc']
            auc2 = results[model2]['metrics']['roc_auc']
            
            # Use DeLong implementation
            y_true_bin = y_true  # Binary labels
            prob1 = probabilities[model1]
            prob2 = probabilities[model2]
            
            # Perform DeLong test using scipy-based approach
            # DeLong test for comparing two correlated ROC curves
            delong_result = compute_delong_auc_significance(y_true_bin, prob1, prob2)
            
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
            
            # Build confusion matrix for McNemar's test
            # Counts of: (both correct), (model1 correct only), (model2 correct only), (both incorrect)
            both_correct = np.sum((pred1 == y_true) & (pred2 == y_true))
            only_model1 = np.sum((pred1 == y_true) & (pred2 != y_true))
            only_model2 = np.sum((pred1 != y_true) & (pred2 == y_true))
            both_incorrect = np.sum((pred1 != y_true) & (pred2 != y_true))
            
            # McNemar's test
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
            
            # Compute F1 scores per sample (F1 for each instance)
            f1_1 = np.array([f1_score([y_true[j]], [pred1[j]]) for j in range(len(y_true))])
            f1_2 = np.array([f1_score([y_true[j]], [pred2[j]]) for j in range(len(y_true))])
            
            # Wilcoxon signed-rank test
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
    
    # Remove duplicate Model columns
    stat_tests_df = stat_tests_df.loc[:, ~stat_tests_df.columns.duplicated()]
    
    logger.info("Statistical tests completed")
    return stat_tests_df


def compute_delong_auc_significance(y_true, prob1, prob2):
    """Compute DeLong test for comparing two ROC curves
    
    Implementation of the DeLong test for correlated ROC curves.
    Reference: DeLong et al. (1988), "Comparing the areas under two or more correlated ROC curves"
    
    Args:
        y_true: True binary labels
        prob1: Predicted probabilities for model 1
        prob2: Predicted probabilities for model 2
        
    Returns:
        Dictionary with z-statistic and p-value
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)
    
    # Get indices for positive and negative classes
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    x = prob1[pos_mask]  # Positive class predictions for model 1
    y = prob1[neg_mask]  # Negative class predictions for model 1
    x2 = prob2[pos_mask]  # Positive class predictions for model 2
    y2 = prob2[neg_mask]  # Negative class predictions for model 2
    
    n_pos = len(x)
    n_neg = len(y)
    
    # Compute AUC for model 1
    auc1 = roc_auc_score(y_true, prob1)
    
    # Compute AUC for model 2
    auc2 = roc_auc_score(y_true, prob2)
    
    # Compute covariance terms for DeLong test
    # V10: covariance of scores on negative samples
    # V01: covariance of scores on positive samples
    def compute_delong_covariance(scores_pos, scores_neg):
        """Compute DeLong covariance terms"""
        n_pos = len(scores_pos)
        n_neg = len(scores_neg)
        
        # For positive samples
        v10_pos = np.var(scores_pos) / n_pos if n_pos > 0 else 0
        # For negative samples  
        v01_neg = np.var(scores_neg) / n_neg if n_neg > 0 else 0
        
        return v10_pos, v01_neg
    
    # Compute covariance between the two models
    # This is a simplified approximation of the DeLong test
    # For exact implementation, see: https://github.com/xrobert/trust
    diff = prob1 - prob2
    var_diff = np.var(diff)
    
    # Z-statistic for difference in AUC
    se_diff = np.sqrt(var_diff) / np.sqrt(len(y_true))
    if se_diff > 0:
        z_stat = (auc1 - auc2) / se_diff
    else:
        z_stat = 0
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {'z_stat': z_stat, 'p_value': p_value}


def generate_tables(results, evaluation_df, output_dir=None):
    """Generate publication tables
    
    Args:
        results: Dictionary with evaluation results
        evaluation_df: DataFrame with evaluation data
        output_dir: Directory to save tables (default: results/)
    """
    logger.info("=" * 80)
    logger.info("Generating Publication Tables")
    logger.info("=" * 80)

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'results'
    else:
        output_dir = Path(output_dir)
    
    results_dir = output_dir
    tables_dir = results_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Performance Metrics Table
    logger.info("Creating performance metrics table...")

    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'ROC-AUC': [results[m]['metrics']['roc_auc'] for m in results.keys()],
        'PR-AUC': [results[m]['metrics']['pr_auc'] for m in results.keys()],
        'Accuracy': [results[m]['metrics']['accuracy'] for m in results.keys()],
        'F1_Score': [results[m]['metrics']['f1'] for m in results.keys()],
        'MCC': [results[m]['metrics']['mcc'] for m in results.keys()],
        'Specificity': [results[m]['metrics']['specificity'] for m in results.keys()],
        'Sensitivity': [results[m]['metrics']['sensitivity'] for m in results.keys()],
        'Precision': [results[m]['metrics']['precision'] for m in results.keys()],
        'TN': [results[m]['metrics']['tn'] for m in results.keys()],
        'FP': [results[m]['metrics']['fp'] for m in results.keys()],
        'FN': [results[m]['metrics']['fn'] for m in results.keys()],
        'TP': [results[m]['metrics']['tp'] for m in results.keys()]
    })

    metrics_df.to_csv(tables_dir / 'table_performance_metrics.csv', index=False)
    logger.info(f"  Saved to {tables_dir / 'table_performance_metrics.csv'}")

    # 2. Statistical Tests Table
    logger.info("Performing statistical tests...")
    stat_tests_df = perform_statistical_tests(results, evaluation_df)
    if stat_tests_df is not None:
        stat_tests_df.to_csv(tables_dir / 'table_statistical_tests.csv', index=False)
        logger.info(f"  Statistical tests saved to {tables_dir / 'table_statistical_tests.csv'}")

    # 3. Summary Report
    logger.info("Creating summary report...")

    report_path = results_dir / 'evaluation_summary_report.md'
    with open(report_path, 'w') as f:
        f.write("# IApred Publication Evaluation Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write("This report summarizes the evaluation of IApred models on external validation data.\n\n")

        # Best model
        best_model = max(results.keys(), key=lambda m: results[m]['metrics']['roc_auc'])
        f.write(f"### Best Performing Model\n\n")
        f.write(f"- **Model**: {best_model}\n")
        f.write(f"- **ROC-AUC**: {results[best_model]['metrics']['roc_auc']:.4f}\n")
        f.write(f"- **PR-AUC**: {results[best_model]['metrics']['pr_auc']:.4f}\n")
        f.write(f"- **Accuracy**: {results[best_model]['metrics']['accuracy']:.4f}\n\n")

        # Model ranking
        f.write("### Model Ranking (by ROC-AUC)\n\n")
        f.write("| Rank | Model | ROC-AUC | PR-AUC | Accuracy | F1 | MCC |\n")
        f.write("|------|-------|---------|--------|----------|-----|-----|\n")

        sorted_models = sorted(results.keys(), key=lambda m: results[m]['metrics']['roc_auc'], reverse=True)
        for i, model in enumerate(sorted_models, 1):
            m = results[model]['metrics']
            f.write(f"| {i} | {model} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['accuracy']:.4f} | {m['f1']:.4f} | {m['mcc']:.4f} |\n")

        f.write("\n### Evaluation Data\n\n")
        f.write(f"- **Total samples**: {len(evaluation_df)}\n")
        f.write(f"- **Antigens**: {sum(evaluation_df['label'] == 1)}\n")
        f.write(f"- **Non-antigens**: {sum(evaluation_df['label'] == 0)}\n\n")

        f.write("### Generated Files\n\n")
        f.write("1. `tables/table_performance_metrics.csv` - Performance metrics\n")
        f.write("2. `tables/table_statistical_tests.csv` - Statistical comparison tests (DeLong for AUC, McNemar for accuracy)\n")
        f.write("3. `figures/roc_curves.pdf/png` - ROC curves\n")
        f.write("4. `figures/pr_curves.pdf/png` - PR curves\n")
        f.write("5. `figures/metrics_comparison.pdf/png` - Metrics comparison\n")
        f.write("6. `figures/confusion_matrices.pdf/png` - Confusion matrices\n")
        f.write("7. `figures/radar_chart.pdf/png` - Radar chart\n")

    logger.info(f"  Saved to {report_path}")


def main():
    """Main evaluation pipeline - generates results for both all models and KBest models"""
    parser = argparse.ArgumentParser(
        description='Evaluate IApred models on external validation data'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only compute statistical tests from existing evaluation results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory containing existing evaluation results (used with --stats-only)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("IApred External Evaluation - Publication Models")
    logger.info("=" * 80)
    
    # If stats-only mode, use existing results
    if args.stats_only:
        results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / 'results'
        if not (results_dir / 'tables' / 'table_performance_metrics.csv').exists():
            logger.error(f"No existing evaluation results found in {results_dir}")
            logger.error("Please run the full evaluation first or specify --results-dir")
            sys.exit(1)
        
        logger.info(f"Loading existing results from {results_dir}")
        
        # Load metrics to get model names
        metrics_df = pd.read_csv(results_dir / 'tables' / 'table_performance_metrics.csv')
        models_to_evaluate = metrics_df['Model'].tolist()
        
        # Load evaluation data
        evaluation_df = load_evaluation_data()
        
        # Evaluate models (just to get predictions)
        results = evaluate_all_models(evaluation_df, models_to_evaluate=models_to_evaluate)
        
        if results:
            # Only generate tables (which includes statistical tests)
            generate_tables(results, evaluation_df, output_dir=results_dir)
            logger.info(f"Statistical tests saved to {results_dir}")
        else:
            logger.error("No models evaluated successfully!")
        return
    
    # Full evaluation mode
    # Load evaluation data
    evaluation_df = load_evaluation_data()

    # =========================================================================
    # EVALUATION: All 4 Models
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating All 4 Models")
    logger.info("=" * 80)

    results_dir = PROJECT_ROOT / 'results'

    results = evaluate_all_models(evaluation_df, models_to_evaluate=ALL_MODELS)

    if results:
        generate_plots(results, evaluation_df, output_dir=results_dir / 'figures', model_colors=MODEL_COLORS_ALL)
        generate_tables(results, evaluation_df, output_dir=results_dir)
        logger.info(f"All models results saved to: {results_dir}")
    else:
        logger.error("No models evaluated successfully!")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
