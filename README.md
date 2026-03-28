# Foundation Models in Bioinformatics: Protein Antigenicity Prediction

**Author:** Sebastian Miles  
**Contact:** smiles@higiene.edu.uy  
**Affiliation:** smilesinformatics.com

---

## Abstract

This work presents a systematic comparison of foundation models against traditional machine learning methods for protein antigenicity prediction. We evaluated five model configurations: TabPFN with SelectKBest feature selection, TabPFN with all features (Default), Random Forest with SelectKBest, XGBoost with SelectKBest, and a pre-trained SVM (IApred) as baseline.

The key finding is that **TabPFN (Default)**, using all 789 features without feature selection, achieved the best overall performance (ROC-AUC: 0.807, Accuracy: 74.1%), outperforming both heavily optimized tree-based methods and the feature-selected TabPFN variant. This suggests foundation models may reduce or eliminate the need for extensive feature engineering in bioinformatics applications.

---

## Results Summary

### Performance on External Validation Data (n=436)

| Model | ROC-AUC | PR-AUC | Accuracy | F1 | MCC | Features Used |
|-------|---------|--------|----------|-----|-----|---------------|
| **TabPFN (Default)** | **0.807** | **0.824** | **74.1%** | **0.751** | **0.481** | 789 (all) |
| XGBoost (SelectKBest) | 0.800 | 0.814 | 74.5% | 0.752 | 0.491 | 405 |
| TabPFN (SelectKBest) | 0.799 | 0.814 | 73.9% | 0.739 | 0.478 | 30 |
| RF (SelectKBest) | 0.798 | 0.817 | 74.1% | 0.748 | 0.481 | 75 |
| IApred (SVM) | 0.782 | 0.800 | 72.2% | 0.731 | 0.445 | 789 |

### Key Findings

1. **TabPFN (Default) achieves best overall performance** with ROC-AUC of 0.807, using all available features without any feature selection. This model required minimal training time (133 seconds) and no hyperparameter tuning.

2. **Feature selection benefits vary by model architecture:**
   - TabPFN with SelectKBest used only 30 features (96% reduction) but lost performance (ROC-AUC dropped from 0.807 to 0.799)
   - XGBoost benefited from feature selection, using 405 features optimally
   - Random Forest achieved best results with just 75 features

3. **Statistical significance (DeLong test for ROC-AUC):**
   - TabPFN (Default) significantly outperforms IApred (SVM) (p = 0.0007)
   - TabPFN (SelectKBest) significantly outperforms IApred (SVM) (p = 0.02)
   - RF (SelectKBest) significantly outperforms IApred (SVM) (p = 0.008)
   - XGBoost (SelectKBest) significantly outperforms IApred (SVM) (p = 0.048)
   - No significant differences between TabPFN (Default), XGBoost, RF, and TabPFN (SelectKBest)

4. **Training efficiency:**
   - TabPFN (Default): 2.2 minutes (no tuning, no feature selection)
   - XGBoost: 4.4 minutes (includes hyperparameter optimization)
   - Random Forest: 13.8 minutes (includes hyperparameter optimization)
   - TabPFN (SelectKBest): 37 minutes (includes k-selection via cross-validation)

### Confusion Matrix Comparison

| Model | True Positives | True Negatives | False Positives | False Negatives |
|-------|---------------|----------------|-----------------|-----------------|
| TabPFN (Default) | 170 | 153 | 61 | 52 |
| TabPFN (SelectKBest) | 161 | 161 | 53 | 61 |
| RF (SelectKBest) | 168 | 155 | 59 | 54 |
| XGBoost (SelectKBest) | 168 | 157 | 57 | 54 |
| IApred (SVM) | 164 | 151 | 63 | 58 |

---

## Methods

### Dataset

- **Training data:** Balanced dataset of antigens and non-antigens from multiple pathogens
- **External validation:** 436 samples (222 antigens, 214 non-antigens) held out from training
- **Evaluation antigens:** Includes proteins from *A. fumigatus*, *H. capsulatum*, *M. bovis*, *S. aureus*, *T. cruzi*, and others
- **Evaluation non-antigens:** Matched set of non-antigenic proteins from same organisms

### Feature Extraction

Each protein sequence was converted to 789 features:
- 9 basic physicochemical properties (length, molecular weight, pI, secondary structure fractions, gravy, aromaticity, instability index)
- 17 additional descriptors (aliphatic index, entropy, hydrophobic moment, charge properties)
- ~50 E-descriptors based on amino acid physicochemical properties
- ~400 ELM motif pattern matches
- 400 dipeptide (2-mer) frequencies

### Feature Selection

SelectKBest with ANOVA F-test was used for feature selection. The optimal number of features (k) was determined via 5-fold cross-validation using the **1-SE rule**: select the smallest k within one standard error of the best cross-validation score. This approach favors simpler models when performance differences are within variance.

| Model | Features After Variance Filter | Optimal k (1-SE Rule) | Reduction |
|-------|-------------------------------|----------------------|-----------|
| TabPFN | 789 | 30 | 96.2% |
| RF | 789 | 75 | 90.5% |
| XGBoost | 789 | 405 | 48.7% |

### Model Training

**Random Forest and XGBoost:**
1. Variance filtering (remove constant features)
2. SelectKBest with optimal k from cross-validation
3. StandardScaler normalization
4. SMOTE for class imbalance
5. 100 iterations of RandomizedSearchCV with 10-fold cross-validation

**TabPFN (SelectKBest):**
1. Variance filtering
2. SelectKBest with optimal k from cross-validation
3. Direct training (no scaling, no SMOTE - TabPFN handles raw features)

**TabPFN (Default):**
1. Variance filtering only
2. Direct training with all remaining features

**IApred (SVM):**
- Pre-trained model downloaded from GitHub
- Uses decision threshold of 0.0 (positive = antigen, negative = non-antigen)

### Statistical Tests

- **DeLong test:** Compare ROC-AUC between models
- **McNemar test:** Compare accuracy between models
- **Wilcoxon signed-rank test:** Compare F1 scores

Significance threshold: α = 0.05

---

## Discussion

The results demonstrate that foundation models, specifically TabPFN, can achieve competitive performance in bioinformatics tasks with substantially less engineering effort. TabPFN (Default) achieved the highest ROC-AUC without any feature selection or hyperparameter tuning, while the tree-based methods required extensive optimization to reach similar performance levels.

The feature selection analysis reveals an interesting pattern: TabPFN performs better with all features, suggesting the model can effectively utilize the full feature space. In contrast, traditional methods benefit from dimensionality reduction, with Random Forest achieving best results with only 75 features (90% reduction).

From a practical standpoint, TabPFN (Default) offers the best trade-off: highest performance, shortest training time, and no tuning required. This makes it particularly suitable for rapid prototyping and applications where computational resources or ML expertise are limited.

The SVM baseline (IApred) remains competitive but is significantly outperformed by modern methods, validating the continued advancement of ML approaches in bioinformatics.

---

## Using the Predictor

Trained models are available for predicting antigenicity of novel protein sequences:

```bash
# Single sequence
python Predictor/predictor.py -s "MKWVTFISLLFLFSSAYSR..."

# From FASTA file
python Predictor/predictor.py -f sequences.fasta --csv results.csv

# Specify model
python Predictor/predictor.py -f sequences.fasta --model-dir models/TabPFN
```

Available models: `TabPFN`, `TabPFN_Default`, `RandomForest`, `XGBoost`, `IApred_SVM`

---

## Installation

```bash
git clone https://github.com/sebamiles/FoundationModelsInBioinformatics.git
cd FoundationModelsInBioinformatics

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# PyTorch with CUDA (recommended for TabPFN)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

---

## Reproducing Results

Run the complete training and evaluation pipeline:

```bash
# Windows
one_click_analysis.bat

# Linux/Mac
chmod +x one_click_analysis.sh
./one_click_analysis.sh
```

This will:
1. Train all five model configurations
2. Evaluate on external validation data
3. Generate performance tables and statistical tests
4. Create publication-ready figures (ROC curves, PR curves, confusion matrices, radar chart)

Output files are saved to `results/tables/` and `results/figures/`.

---

## Citation

```
Miles, S. Foundation Models in Bioinformatics: Protein Antigenicity Prediction.
GitHub: https://github.com/sebamiles/FoundationModelsInBioinformatics
```

---

## License

Research code—use it, modify it, just don't blame me if it breaks.

---

## Support

For questions, issues, or collaboration: **smiles@higiene.edu.uy**
