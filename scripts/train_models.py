#!/usr/bin/env python3
"""
Master Training Script for IApred Publication

Trains 4 models with SelectKBest feature selection:
1. RandomForest (SelectKBest + hyperparameter optimization)
2. XGBoost (SelectKBest + hyperparameter optimization)
3. TabPFN (SelectKBest)
4. IApred_SVM (downloaded from GitHub)

Expected runtime: ~3-4 hours total (including k-selection and hyperparameter optimization)
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import StratifiedKFold

# Add Predictor to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Predictor'))

from data_loader import load_training_data
from functions_for_training import sequences_to_vectors

# Setup
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def download_iapred_svm():
    """Download IApred SVM model from GitHub"""
    logger.info("=" * 80)
    logger.info("Downloading IApred SVM Model")
    logger.info("=" * 80)

    try:
        # Import download script
        download_script = PROJECT_ROOT / 'scripts' / 'download_iapred_model.py'
        if download_script.exists():
            # Run download script
            import subprocess
            result = subprocess.run([
                sys.executable,
                str(download_script),
                '-o', str(MODELS_DIR / 'IApred_SVM')
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("IApred SVM model downloaded successfully")
                return True
            else:
                logger.error(f"Failed to download IApred SVM model: {result.stderr}")
                return False
        else:
            logger.error("Download script not found")
            return False
    except Exception as e:
        logger.error(f"Error downloading IApred SVM model: {e}")
        return False


def train_randomforest(X_train, y_train, X_val, y_val, feature_names):
    """Train RandomForest with SelectKBest feature selection followed by hyperparameter optimization"""
    logger.info("=" * 80)
    logger.info("Training RandomForest (SelectKBest + Optimization)")
    logger.info("=" * 80)

    model_dir = MODELS_DIR / 'RandomForest'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Start timing from here to include all preprocessing and feature selection
    total_start_time = time.time()

    # Variance threshold
    variance_selector = VarianceThreshold(threshold=0.0)
    X_train_var = variance_selector.fit_transform(X_train)
    X_val_var = variance_selector.transform(X_val)

    # Feature names after variance filtering
    feature_names_var = [feature_names[i] for i in variance_selector.get_support(indices=True)]
    logger.info(f"Features after variance filtering: {len(feature_names_var)}")

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_var)
    X_val_scaled = scaler.transform(X_val_var)

    # Step 1: Find optimal k using cross-validation (every 15 K)
    logger.info("Step 1: Finding optimal number of features (k)...")
    n_features = X_train_scaled.shape[1]
    min_k = max(10, int(n_features * 0.02))  # At least 10 or 2% of features
    max_k = int(n_features * 0.95)  # Up to 95% of features

    # Generate k values: every 15 K
    k_values = list(range(min_k, max_k + 1, 15))
    k_values = sorted(set(k_values))  # Remove duplicates and sort

    logger.info(f"Testing k values: {k_values}")

    best_k = k_values[0]
    best_k_score = 0
    k_scores = {}   # Store all k mean scores
    k_stds = {}     # Store all k std scores

    for k in k_values:
        logger.info(f"  Testing k={k}...")
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_k = selector.fit_transform(X_train_scaled, y_train)

        # Quick cross-validation with default RF
        rf_temp = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100)
        cv_scores = []
        for train_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_k, y_train):
            X_tr, X_vl = X_train_k[train_idx], X_train_k[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]
            # Apply SMOTE only to training fold
            smote = SMOTE(random_state=42)
            X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
            rf_temp.fit(X_tr_smote, y_tr_smote)
            y_vl_proba = rf_temp.predict_proba(X_vl)[:, 1]
            cv_scores.append(roc_auc_score(y_vl, y_vl_proba))

        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        k_scores[k] = mean_score
        k_stds[k] = std_score
        logger.info(f"    k={k}: CV ROC-AUC = {mean_score:.4f} ± {std_score:.4f}")

    # 1-SE Rule: pick smallest k within 1 std-error of the global best
    k_arr = np.array(list(k_scores.keys()))
    means = np.array(list(k_scores.values()))
    stds = np.array(list(k_stds.values()))

    best_idx = np.argmax(means)
    threshold = means[best_idx] - stds[best_idx]  # best - 1·SE
    candidates = np.where(means >= threshold)[0]
    selected = candidates[0]  # smallest k that passes

    best_k = int(k_arr[selected])
    best_k_score = means[selected]

    logger.info(f"Global best k: {k_arr[best_idx]} (ROC-AUC={means[best_idx]:.4f})")
    logger.info(f"1-SE threshold: {threshold:.4f}")
    logger.info(f"Selected k (1-SE rule): {best_k} (ROC-AUC={best_k_score:.4f}) "
                f"— {100*(1 - best_k/k_arr[best_idx]):.1f}% fewer features")

    # Step 2: Apply SelectKBest with optimal k
    logger.info(f"Step 2: Applying SelectKBest with k={best_k}...")
    feature_selector = SelectKBest(score_func=f_classif, k=best_k)
    X_train_kbest = feature_selector.fit_transform(X_train_scaled, y_train)
    X_val_kbest = feature_selector.transform(X_val_scaled)

    # Get selected feature names
    selected_indices = feature_selector.get_support(indices=True)
    feature_names_kbest = [feature_names_var[i] for i in selected_indices]
    logger.info(f"Selected {len(feature_names_kbest)} features")

    # Step 3: Apply SMOTE on selected features
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_kbest, y_train)
    logger.info(f"Training samples after SMOTE: {len(y_train_smote)}")

    # Step 4: Hyperparameter search
    logger.info("Step 3: Hyperparameter optimization...")
    param_dist = {
        'n_estimators': randint(100, 1000),
        'max_depth': [None] + list(range(10, 50, 10)),
        'max_features': uniform(0.1, 0.9),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=100,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42
    )

    start_time = time.time()
    random_search.fit(X_train_smote, y_train_smote)
    elapsed_time = time.time() - start_time

    best_model = random_search.best_estimator_
    logger.info(f"Best parameters: {random_search.best_params_}")
    logger.info(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")
    logger.info(f"Hyperparameter optimization time: {elapsed_time:.2f} seconds")

    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val_kbest)
    y_val_proba = best_model.predict_proba(X_val_kbest)[:, 1]

    metrics = {
        'roc_auc': roc_auc_score(y_val, y_val_proba),
        'pr_auc': average_precision_score(y_val, y_val_proba),
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'mcc': matthews_corrcoef(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'cm': confusion_matrix(y_val, y_val_pred)
    }

    logger.info(f"Validation metrics: {metrics}")

    # Save model
    joblib.dump(best_model, model_dir / 'IApred_RandomForest.joblib')
    joblib.dump(variance_selector, model_dir / 'IApred_variance_selector.joblib')
    joblib.dump(scaler, model_dir / 'IApred_scaler.joblib')
    joblib.dump(feature_selector, model_dir / 'IApred_feature_selector.joblib')
    # Save ORIGINAL feature names (before variance filtering) for alignment
    joblib.dump(feature_names, model_dir / 'IApred_all_feature_names.joblib')
    # Also save selected feature names for reference
    joblib.dump(feature_names_kbest, model_dir / 'IApred_feature_names.joblib')

    # Calculate total training time (including preprocessing and feature selection)
    total_elapsed_time = time.time() - total_start_time

    # Save training metrics
    training_metrics = {
        'best_params': random_search.best_params_,
        'best_cv_score': random_search.best_score_,
        'cv_results': random_search.cv_results_,
        'training_time': elapsed_time,
        'total_training_time': total_elapsed_time,
        'optimal_k': best_k,
        'k_selection_score': best_k_score,
        'k_scores': k_scores,  # All k values and their scores
        'k_stds': k_stds,      # All k values and their stds
        'validation_metrics': metrics,
        'n_features_selected': len(feature_names_kbest),
        'total_features': len(feature_names_var)
    }
    joblib.dump(training_metrics, model_dir / 'IApred_training_metrics.joblib')

    logger.info(f"Model saved to {model_dir}")
    logger.info(f"Total training time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
    return model_dir


def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """Train XGBoost with SelectKBest feature selection followed by hyperparameter optimization"""
    logger.info("=" * 80)
    logger.info("Training XGBoost (SelectKBest + Optimization)")
    logger.info("=" * 80)

    model_dir = MODELS_DIR / 'XGBoost'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Start timing from here to include all preprocessing and feature selection
    total_start_time = time.time()

    # Variance threshold
    variance_selector = VarianceThreshold(threshold=0.0)
    X_train_var = variance_selector.fit_transform(X_train)
    X_val_var = variance_selector.transform(X_val)

    # Feature names after variance filtering
    feature_names_var = [feature_names[i] for i in variance_selector.get_support(indices=True)]
    logger.info(f"Features after variance filtering: {len(feature_names_var)}")

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_var)
    X_val_scaled = scaler.transform(X_val_var)

    # Step 1: Find optimal k using cross-validation (every 15 K)
    logger.info("Step 1: Finding optimal number of features (k)...")
    n_features = X_train_scaled.shape[1]
    min_k = max(10, int(n_features * 0.02))  # At least 10 or 2% of features
    max_k = int(n_features * 0.95)  # Up to 95% of features

    # Generate k values: every 15 K
    k_values = list(range(min_k, max_k + 1, 15))
    k_values = sorted(set(k_values))  # Remove duplicates and sort

    logger.info(f"Testing k values: {k_values}")

    best_k = k_values[0]
    best_k_score = 0
    k_scores = {}   # Store all k mean scores
    k_stds = {}     # Store all k std scores

    for k in k_values:
        logger.info(f"  Testing k={k}...")
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_k = selector.fit_transform(X_train_scaled, y_train)

        # Quick cross-validation with default XGBoost
        xgb_temp = xgb.XGBClassifier(random_state=42, n_jobs=-1, n_estimators=100,
                                      eval_metric='auc', use_label_encoder=False, verbosity=0)
        cv_scores = []
        for train_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_k, y_train):
            X_tr, X_vl = X_train_k[train_idx], X_train_k[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]
            # Apply SMOTE only to training fold
            smote = SMOTE(random_state=42)
            X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
            xgb_temp.fit(X_tr_smote, y_tr_smote, verbose=False)
            y_vl_proba = xgb_temp.predict_proba(X_vl)[:, 1]
            cv_scores.append(roc_auc_score(y_vl, y_vl_proba))

        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        k_scores[k] = mean_score
        k_stds[k] = std_score
        logger.info(f"    k={k}: CV ROC-AUC = {mean_score:.4f} ± {std_score:.4f}")

    # 1-SE Rule: pick smallest k within 1 std-error of the global best
    k_arr = np.array(list(k_scores.keys()))
    means = np.array(list(k_scores.values()))
    stds = np.array(list(k_stds.values()))

    best_idx = np.argmax(means)
    threshold = means[best_idx] - stds[best_idx]  # best - 1·SE
    candidates = np.where(means >= threshold)[0]
    selected = candidates[0]  # smallest k that passes

    best_k = int(k_arr[selected])
    best_k_score = means[selected]

    logger.info(f"Global best k: {k_arr[best_idx]} (ROC-AUC={means[best_idx]:.4f})")
    logger.info(f"1-SE threshold: {threshold:.4f}")
    logger.info(f"Selected k (1-SE rule): {best_k} (ROC-AUC={best_k_score:.4f}) "
                f"— {100*(1 - best_k/k_arr[best_idx]):.1f}% fewer features")

    # Step 2: Apply SelectKBest with optimal k
    logger.info(f"Step 2: Applying SelectKBest with k={best_k}...")
    feature_selector = SelectKBest(score_func=f_classif, k=best_k)
    X_train_kbest = feature_selector.fit_transform(X_train_scaled, y_train)
    X_val_kbest = feature_selector.transform(X_val_scaled)

    # Get selected feature names
    selected_indices = feature_selector.get_support(indices=True)
    feature_names_kbest = [feature_names_var[i] for i in selected_indices]
    logger.info(f"Selected {len(feature_names_kbest)} features")

    # Step 3: Apply SMOTE on selected features
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_kbest, y_train)
    logger.info(f"Training samples after SMOTE: {len(y_train_smote)}")

    # Step 4: Hyperparameter search
    logger.info("Step 3: Hyperparameter optimization...")
    param_dist = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(3, 20),
        'learning_rate': uniform(0.001, 0.1),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'gamma': uniform(0, 5),
        'reg_alpha': loguniform(1e-5, 100),
        'reg_lambda': loguniform(1e-5, 100)
    }

    xgb_model = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='auc',
        use_label_encoder=False
    )
    random_search = RandomizedSearchCV(
        xgb_model, param_distributions=param_dist, n_iter=100,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42
    )

    start_time = time.time()
    random_search.fit(X_train_smote, y_train_smote)
    elapsed_time = time.time() - start_time

    best_model = random_search.best_estimator_
    logger.info(f"Best parameters: {random_search.best_params_}")
    logger.info(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")
    logger.info(f"Hyperparameter optimization time: {elapsed_time:.2f} seconds")

    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val_kbest)
    y_val_proba = best_model.predict_proba(X_val_kbest)[:, 1]

    metrics = {
        'roc_auc': roc_auc_score(y_val, y_val_proba),
        'pr_auc': average_precision_score(y_val, y_val_proba),
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'mcc': matthews_corrcoef(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'cm': confusion_matrix(y_val, y_val_pred)
    }

    logger.info(f"Validation metrics: {metrics}")

    # Save model
    joblib.dump(best_model, model_dir / 'IApred_XGBoost.joblib')
    joblib.dump(variance_selector, model_dir / 'IApred_variance_selector.joblib')
    joblib.dump(scaler, model_dir / 'IApred_scaler.joblib')
    joblib.dump(feature_selector, model_dir / 'IApred_feature_selector.joblib')
    # Save ORIGINAL feature names (before variance filtering) for alignment
    joblib.dump(feature_names, model_dir / 'IApred_all_feature_names.joblib')
    # Also save selected feature names for reference
    joblib.dump(feature_names_kbest, model_dir / 'IApred_feature_names.joblib')

    # Calculate total training time (including preprocessing and feature selection)
    total_elapsed_time = time.time() - total_start_time

    # Save training metrics
    training_metrics = {
        'best_params': random_search.best_params_,
        'best_cv_score': random_search.best_score_,
        'cv_results': random_search.cv_results_,
        'training_time': elapsed_time,
        'total_training_time': total_elapsed_time,
        'optimal_k': best_k,
        'k_selection_score': best_k_score,
        'k_scores': k_scores,  # All k values and their scores
        'k_stds': k_stds,      # All k values and their stds
        'validation_metrics': metrics,
        'n_features_selected': len(feature_names_kbest),
        'total_features': len(feature_names_var)
    }
    joblib.dump(training_metrics, model_dir / 'IApred_training_metrics.joblib')

    logger.info(f"Model saved to {model_dir}")
    logger.info(f"Total training time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
    return model_dir


def train_tabpfn(X_train, y_train, X_val, y_val, feature_names):
    """Train TabPFN with SelectKBest feature selection"""
    logger.info("=" * 80)
    logger.info("Training TabPFN (SelectKBest)")
    logger.info("=" * 80)

    model_dir = MODELS_DIR / 'TabPFN'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Start timing from here to include all preprocessing and feature selection
    total_start_time = time.time()

    # For TabPFN, we use variance filtering only (no scaling, no SMOTE)
    variance_selector = VarianceThreshold(threshold=0.0)
    X_train_var = variance_selector.fit_transform(X_train)
    X_val_var = variance_selector.transform(X_val)

    # Feature names after variance filtering
    feature_names_var = [feature_names[i] for i in variance_selector.get_support(indices=True)]
    logger.info(f"Features after variance filtering: {len(feature_names_var)}")

    # Step 1: Find optimal k using cross-validation (every 15 K)
    logger.info("Step 1: Finding optimal number of features (k)...")
    n_features = X_train_var.shape[1]
    min_k = max(10, int(n_features * 0.02))  # At least 10 or 2% of features
    max_k = int(n_features * 0.95)  # Up to 95% of features

    # Generate k values: every 15 K
    k_values = list(range(min_k, max_k + 1, 15))
    k_values = sorted(set(k_values))  # Remove duplicates and sort

    logger.info(f"Testing k values: {k_values}")

    best_k = k_values[0]
    best_k_score = 0
    k_scores = {}   # Store all k mean scores
    k_stds = {}     # Store all k std scores

    for k in k_values:
        logger.info(f"  Testing k={k}...")
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_k = selector.fit_transform(X_train_var, y_train)

        # Quick cross-validation with TabPFN
        from tabpfn import TabPFNClassifier
        tabpfn_temp = TabPFNClassifier(
            n_estimators=8,  # Fewer estimators for faster CV
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        cv_scores = []
        for train_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_k, y_train):
            X_tr, X_vl = X_train_k[train_idx], X_train_k[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]
            # TabPFN doesn't use SMOTE
            tabpfn_temp.fit(X_tr, y_tr)
            y_vl_proba = tabpfn_temp.predict_proba(X_vl)[:, 1]
            cv_scores.append(roc_auc_score(y_vl, y_vl_proba))

        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        k_scores[k] = mean_score
        k_stds[k] = std_score
        logger.info(f"    k={k}: CV ROC-AUC = {mean_score:.4f} ± {std_score:.4f}")

    # 1-SE Rule: pick smallest k within 1 std-error of the global best
    k_arr = np.array(list(k_scores.keys()))
    means = np.array(list(k_scores.values()))
    stds = np.array(list(k_stds.values()))

    best_idx = np.argmax(means)
    threshold = means[best_idx] - stds[best_idx]  # best - 1·SE
    candidates = np.where(means >= threshold)[0]
    selected = candidates[0]  # smallest k that passes

    best_k = int(k_arr[selected])
    best_k_score = means[selected]

    logger.info(f"Global best k: {k_arr[best_idx]} (ROC-AUC={means[best_idx]:.4f})")
    logger.info(f"1-SE threshold: {threshold:.4f}")
    logger.info(f"Selected k (1-SE rule): {best_k} (ROC-AUC={best_k_score:.4f}) "
                f"— {100*(1 - best_k/k_arr[best_idx]):.1f}% fewer features")

    # Step 2: Apply SelectKBest with optimal k
    logger.info(f"Step 2: Applying SelectKBest with k={best_k}...")
    feature_selector = SelectKBest(score_func=f_classif, k=best_k)
    X_train_kbest = feature_selector.fit_transform(X_train_var, y_train)
    X_val_kbest = feature_selector.transform(X_val_var)

    # Get selected feature names
    selected_indices = feature_selector.get_support(indices=True)
    feature_names_kbest = [feature_names_var[i] for i in selected_indices]
    logger.info(f"Selected {len(feature_names_kbest)} features")

    # Train TabPFN with selected features
    from tabpfn import TabPFNClassifier

    start_time = time.time()
    model = TabPFNClassifier(
        n_estimators=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.fit(X_train_kbest, y_train)
    elapsed_time = time.time() - start_time

    logger.info(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Evaluate on validation set
    y_val_pred = model.predict(X_val_kbest)
    y_val_proba = model.predict_proba(X_val_kbest)[:, 1]

    metrics = {
        'roc_auc': roc_auc_score(y_val, y_val_proba),
        'pr_auc': average_precision_score(y_val, y_val_proba),
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'mcc': matthews_corrcoef(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'cm': confusion_matrix(y_val, y_val_pred)
    }

    logger.info(f"Validation metrics: {metrics}")

    # Save model
    joblib.dump(model, model_dir / 'IApred_TabPFN.joblib')
    joblib.dump(variance_selector, model_dir / 'IApred_variance_selector.joblib')
    joblib.dump(feature_selector, model_dir / 'IApred_feature_selector.joblib')
    # Save ORIGINAL feature names (before variance filtering) for alignment
    joblib.dump(feature_names, model_dir / 'IApred_all_feature_names.joblib')
    # Also save selected feature names for reference
    joblib.dump(feature_names_kbest, model_dir / 'IApred_feature_names.joblib')

    # Calculate total training time (including preprocessing and feature selection)
    total_elapsed_time = time.time() - total_start_time

    # Save training metrics
    training_metrics = {
        'training_time_seconds': elapsed_time,
        'training_time_minutes': elapsed_time / 60,
        'total_training_time_seconds': total_elapsed_time,
        'total_training_time_minutes': total_elapsed_time / 60,
        'optimal_k': best_k,
        'k_selection_score': best_k_score,
        'k_scores': k_scores,  # All k values and their scores
        'k_stds': k_stds,      # All k values and their stds
        'validation_metrics': metrics,
        'n_features_selected': len(feature_names_kbest),
        'total_features': len(feature_names_var),
        'n_samples': len(y_train)
    }
    joblib.dump(training_metrics, model_dir / 'IApred_training_metrics.joblib')

    logger.info(f"Model saved to {model_dir}")
    logger.info(f"Total training time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
    return model_dir


def train_tabpfn_default(X_train, y_train, X_val, y_val, feature_names):
    """Train TabPFN without any feature selection (uses all features after variance filtering)"""
    logger.info("=" * 80)
    logger.info("Training TabPFN_Default (No Feature Selection)")
    logger.info("=" * 80)

    model_dir = MODELS_DIR / 'TabPFN_Default'
    model_dir.mkdir(parents=True, exist_ok=True)

    # Start timing from here to include all preprocessing
    total_start_time = time.time()

    # For TabPFN, we use variance filtering only (no scaling, no SMOTE)
    variance_selector = VarianceThreshold(threshold=0.0)
    X_train_var = variance_selector.fit_transform(X_train)
    X_val_var = variance_selector.transform(X_val)

    # Feature names after variance filtering
    feature_names_var = [feature_names[i] for i in variance_selector.get_support(indices=True)]
    logger.info(f"Features after variance filtering: {len(feature_names_var)}")
    logger.info(f"Using all {len(feature_names_var)} features (no SelectKBest)")

    # Train TabPFN with all variance-filtered features (no feature selection)
    from tabpfn import TabPFNClassifier

    start_time = time.time()
    model = TabPFNClassifier(
        n_estimators=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.fit(X_train_var, y_train)
    elapsed_time = time.time() - start_time

    logger.info(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Evaluate on validation set
    y_val_pred = model.predict(X_val_var)
    y_val_proba = model.predict_proba(X_val_var)[:, 1]

    metrics = {
        'roc_auc': roc_auc_score(y_val, y_val_proba),
        'pr_auc': average_precision_score(y_val, y_val_proba),
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'mcc': matthews_corrcoef(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'cm': confusion_matrix(y_val, y_val_pred)
    }

    logger.info(f"Validation metrics: {metrics}")

    # Save model
    joblib.dump(model, model_dir / 'IApred_TabPFN_Default.joblib')
    joblib.dump(variance_selector, model_dir / 'IApred_variance_selector.joblib')
    # Save ORIGINAL feature names (before variance filtering) for alignment
    joblib.dump(feature_names, model_dir / 'IApred_all_feature_names.joblib')
    # Also save variance-filtered feature names for reference
    joblib.dump(feature_names_var, model_dir / 'IApred_feature_names.joblib')

    # Calculate total training time (including preprocessing)
    total_elapsed_time = time.time() - total_start_time

    # Save training metrics
    training_metrics = {
        'training_time_seconds': elapsed_time,
        'training_time_minutes': elapsed_time / 60,
        'total_training_time_seconds': total_elapsed_time,
        'total_training_time_minutes': total_elapsed_time / 60,
        'validation_metrics': metrics,
        'n_features_used': len(feature_names_var),
        'total_features': len(feature_names_var),
        'n_samples': len(y_train),
        'feature_selection': 'None (all variance-filtered features used)'
    }
    joblib.dump(training_metrics, model_dir / 'IApred_training_metrics.joblib')

    logger.info(f"Model saved to {model_dir}")
    logger.info(f"Total training time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
    return model_dir


def main():
    """Main training pipeline"""
    logger.info("=" * 80)
    logger.info("IApred Training Pipeline - Publication Models")
    logger.info("=" * 80)

    # Load training data
    logger.info("\nLoading training data...")
    from data_loader import load_training_data
    all_sequences, labels, antigen_files, non_antigen_files = load_training_data()

    # Convert labels to binary (antigen=1, non-antigen=0)
    y = np.array([1 if label == 'antigen' else 0 for label in labels])
    sequences = all_sequences
    X = None  # Will be set after feature extraction

    logger.info(f"Loaded {len(y)} samples: {sum(y==1)} antigens, {sum(y==0)} non-antigens")

    # Extract features
    logger.info("\nExtracting features...")
    X, feature_names, failed_indices = sequences_to_vectors(sequences)
    logger.info(f"Extracted {len(feature_names)} features")

    # Split data
    logger.info("\nSplitting data (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Training: {len(y_train)}, Validation: {len(y_val)}")

    # Helper function to check if model already exists
    def model_exists(model_name):
        """Check if model directory exists and has model files"""
        model_dir = MODELS_DIR / model_name
        if not model_dir.exists():
            return False
        # Check for model files (any IApred_*.joblib except auxiliary files)
        model_files = list(model_dir.glob('IApred_*.joblib'))
        for f in model_files:
            if 'variance_selector' not in f.name and 'feature_names' not in f.name:
                return True
        return False

    # Train models
    results = {}

    # 1. RandomForest (SelectKBest)
    if model_exists('RandomForest'):
        logger.info("\nRandomForest model already exists, skipping training...")
        results['RandomForest'] = str(MODELS_DIR / 'RandomForest')
    else:
        try:
            model_dir = train_randomforest(X_train, y_train, X_val, y_val, feature_names)
            results['RandomForest'] = str(model_dir)
        except Exception as e:
            logger.error(f"RandomForest training failed: {e}")

    # 2. XGBoost (SelectKBest)
    if model_exists('XGBoost'):
        logger.info("\nXGBoost model already exists, skipping training...")
        results['XGBoost'] = str(MODELS_DIR / 'XGBoost')
    else:
        try:
            model_dir = train_xgboost(X_train, y_train, X_val, y_val, feature_names)
            results['XGBoost'] = str(model_dir)
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")

    # 3. TabPFN (SelectKBest)
    if model_exists('TabPFN'):
        logger.info("\nTabPFN model already exists, skipping training...")
        results['TabPFN'] = str(MODELS_DIR / 'TabPFN')
    else:
        try:
            model_dir = train_tabpfn(X_train, y_train, X_val, y_val, feature_names)
            results['TabPFN'] = str(model_dir)
        except Exception as e:
            logger.error(f"TabPFN training failed: {e}")

    # 4. TabPFN_Default (No Feature Selection)
    if model_exists('TabPFN_Default'):
        logger.info("\nTabPFN_Default model already exists, skipping training...")
        results['TabPFN_Default'] = str(MODELS_DIR / 'TabPFN_Default')
    else:
        try:
            model_dir = train_tabpfn_default(X_train, y_train, X_val, y_val, feature_names)
            results['TabPFN_Default'] = str(model_dir)
        except Exception as e:
            logger.error(f"TabPFN_Default training failed: {e}")

    # 5. IApred SVM (download from GitHub)
    if model_exists('IApred_SVM'):
        logger.info("\nIApred_SVM model already exists, skipping download...")
        results['IApred_SVM'] = str(MODELS_DIR / 'IApred_SVM')
    else:
        try:
            logger.info("\nDownloading IApred SVM model from GitHub...")
            if download_iapred_svm():
                results['IApred_SVM'] = str(MODELS_DIR / 'IApred_SVM')
            else:
                logger.error("Failed to download IApred SVM model")
        except Exception as e:
            logger.error(f"IApred SVM download failed: {e}")

    # Save summary
    summary_file = RESULTS_DIR / 'training_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("IApred Training Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Models Trained:\n")
        for model_name, model_path in results.items():
            f.write(f"  {model_name}: {model_path}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED MODEL INFORMATION\n")
        f.write("=" * 80 + "\n\n")

        # Add detailed information for each model
        for model_name, model_path in results.items():
            model_dir = Path(model_path)
            metrics_file = model_dir / 'IApred_training_metrics.joblib'

            f.write("-" * 80 + "\n")
            f.write(f"MODEL: {model_name}\n")
            f.write("-" * 80 + "\n")

            if metrics_file.exists():
                try:
                    metrics = joblib.load(metrics_file)

                    # Training time - use total_training_time if available, otherwise fall back to training_time
                    if 'total_training_time' in metrics:
                        train_time = metrics['total_training_time']
                        f.write(f"Total Training Time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)\n")
                    elif 'total_training_time_seconds' in metrics:
                        train_time = metrics['total_training_time_seconds']
                        f.write(f"Total Training Time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)\n")
                    elif 'training_time' in metrics:
                        train_time = metrics['training_time']
                        f.write(f"Training Time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)\n")
                    elif 'training_time_seconds' in metrics:
                        train_time = metrics['training_time_seconds']
                        f.write(f"Training Time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)\n")

                    # Number of features
                    if 'total_features' in metrics:
                        f.write(f"Total Features (after variance filter): {metrics['total_features']}\n")
                    if 'n_features_selected' in metrics:
                        f.write(f"Features Selected: {metrics['n_features_selected']}\n")
                    elif 'n_features' in metrics:
                        f.write(f"Features Used: {metrics['n_features']}\n")

                    # Best k for KBest models
                    if 'optimal_k' in metrics:
                        f.write(f"\n--- Feature Selection (SelectKBest - 1-SE Rule) ---\n")
                        f.write(f"Optimal k: {metrics['optimal_k']}\n")
                        f.write(f"Best k CV Score (ROC-AUC): {metrics['k_selection_score']:.4f}\n")
                        if 'k_scores' in metrics:
                            f.write(f"All k values tested: {list(metrics['k_scores'].keys())}\n")
                            f.write(f"All k scores: {[f'{v:.4f}' for v in metrics['k_scores'].values()]}\n")
                        if 'k_stds' in metrics:
                            f.write(f"All k stds: {[f'{v:.4f}' for v in metrics['k_stds'].values()]}\n")

                    # Best parameters for models with hyperparameter optimization
                    if 'best_params' in metrics:
                        f.write(f"\n--- Hyperparameter Optimization ---\n")
                        f.write(f"Best CV Score (ROC-AUC): {metrics['best_cv_score']:.4f}\n")
                        f.write(f"Best Parameters:\n")
                        for param, value in metrics['best_params'].items():
                            f.write(f"  {param}: {value}\n")

                    # Validation metrics
                    if 'validation_metrics' in metrics:
                        val_metrics = metrics['validation_metrics']
                        f.write(f"\n--- Validation Metrics ---\n")
                        f.write(f"ROC-AUC: {val_metrics['roc_auc']:.4f}\n")
                        f.write(f"PR-AUC: {val_metrics['pr_auc']:.4f}\n")
                        f.write(f"Accuracy: {val_metrics['accuracy']:.4f}\n")
                        f.write(f"F1 Score: {val_metrics['f1']:.4f}\n")
                        f.write(f"MCC: {val_metrics['mcc']:.4f}\n")
                        f.write(f"Precision: {val_metrics['precision']:.4f}\n")
                        f.write(f"Recall: {val_metrics['recall']:.4f}\n")

                    f.write("\n")

                except Exception as e:
                    f.write(f"Error loading metrics: {e}\n\n")
            else:
                f.write(f"No metrics file found\n\n")

        f.write("=" * 80 + "\n")
        f.write("Training complete!\n")
        f.write("=" * 80 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
