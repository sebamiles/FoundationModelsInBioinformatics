# Foundation Models in Bioinformatics: Protein Antigenicity Prediction

This project came out of our work exploring how foundation models and traditional machine learning approaches stack up against each other for predicting protein antigenicity. We built a complete pipeline that trains multiple models, evaluates them on external data, and spits out publication-ready results.

Part of [smilesinformatics.com](https://smilesinformatics.com)

## What This Does

The pipeline takes protein sequences and predicts whether they're antigens or not. We're comparing four different approaches:

- **Random Forest** - Classic ensemble method with full hyperparameter optimization
- **XGBoost** - Gradient boosting with extensive tuning
- **TabPFN** - A foundation model for tabular data (the interesting one)
- **IApred (SVM)** - A pre-trained SVM we downloaded for comparison

All models use around 787 features extracted from each sequence: physicochemical properties, dipeptide frequencies, ELM motif matches, and E-descriptors. The feature selection uses SelectKBest with the 1-SE rule to find the sweet spot between performance and complexity.

## Repository Layout

```
FoundationModelsInBioinformatics/
├── data/                      # Training and evaluation datasets
│   ├── antigens/              # Training antigens (FASTA)
│   ├── non-antigens/          # Training non-antigens (FASTA)
│   ├── evaluation_antigens/   # External validation antigens
│   ├── evaluation_non_antigens/
│   └── protein_motifs.txt     # ELM patterns for feature extraction
├── Predictor/                 # Core prediction module
│   ├── data_loader.py
│   ├── functions_for_training.py
│   ├── predictor.py           # CLI for single/batch predictions
│   └── requirements.txt
├── scripts/                   # Pipeline scripts
│   ├── train_models.py        # Train all models
│   ├── evaluate_models.py     # Evaluate and generate figures
│   ├── download_iapred_model.py
│   └── compute_statistics.py
├── models/                    # Trained models (generated)
├── results/                   # Output figures and tables (generated)
├── FinalPaper/                # Manuscript files (not tracked)
├── requirements.txt
├── train_pipeline.bat/sh      # Training wrappers
├── evaluate_pipeline.bat/sh   # Evaluation wrappers
└── one_click_analysis.bat/sh  # Full pipeline
```

## Getting Started

### Requirements

- Python 3.9-3.11
- 16GB RAM recommended
- CUDA 11.8+ if you want TabPFN to run in reasonable time (optional, but GPU makes a huge difference)

### Installation

```bash
git clone https://github.com/sebamiles/FoundationModelsInBioinformatics.git
cd FoundationModelsInBioinformatics

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Quick Test

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from tabpfn import TabPFNClassifier; print('TabPFN ready')"
```

## Running the Pipeline

### One-Click Everything

If you just want to run the whole thing:

**Windows:**
```bash
one_click_analysis.bat
```

**Linux/Mac:**
```bash
chmod +x one_click_analysis.sh
./one_click_analysis.sh
```

This trains all models, evaluates them, and generates all figures and tables.

### Step by Step

**Training:**
```bash
# Windows
train_pipeline.bat

# Linux/Mac
chmod +x train_pipeline.sh
./train_pipeline.sh

# Or directly
python scripts/train_models.py
```

Training takes a while. On CPU, expect a few hours. With GPU, TabPFN finishes in minutes but Random Forest and XGBoost still need time for their hyperparameter searches.

**Evaluation:**
```bash
# Windows
evaluate_pipeline.bat

# Linux/Mac
chmod +x evaluate_pipeline.sh
./evaluate_pipeline.sh

# Or directly
python scripts/evaluate_models.py
```

This loads all trained models, runs them on the external validation sets, and generates ROC curves, PR curves, confusion matrices, and comparison tables in `results/`.

## Using the Predictor

Once you have trained models (or if you download the pre-trained ones), you can make predictions:

```bash
# Single sequence
python Predictor/predictor.py -s "MKWVTFISLLFLFSSAYSR..."

# From FASTA file
python Predictor/predictor.py -f sequences.fasta --csv results.csv

# Use a specific model
python Predictor/predictor.py -f sequences.fasta --model-dir models/TabPFN
```

The output gives you a score (higher = more antigenic) and a classification based on the model's threshold.

## What the Models Actually Do

**Random Forest and XGBoost** go through the full treatment: variance filtering, SelectKBest feature selection with cross-validation to find optimal k, SMOTE for class imbalance, then 100 iterations of random search with 10-fold CV. They're slow but thorough.

**TabPFN** is the foundation model here. It's a transformer trained on synthetic tabular data that can be fine-tuned on new datasets with just forward passes. We use it with SelectKBest but skip SMOTE and scaling since TabPFN handles raw features better. The interesting part is how it compares to the heavily optimized tree methods.

**IApred (SVM)** is a pre-trained model we include for reference. It uses a threshold of 0.0 on the decision function rather than probabilities.

## Feature Extraction

Each sequence gets converted into ~787 features:

- 9 basic properties (length, molecular weight, pI, secondary structure fractions, gravy, aromaticity, instability index)
- 17 additional descriptors (aliphatic index, entropy, hydrophobic moment, charge properties)
- ~50 E-descriptors based on amino acid physicochemical properties
- ~400 motif features from the ELM database
- 400 dipeptide frequencies

Variance filtering removes constant features, then SelectKBest picks the most informative subset using ANOVA F-test.

## Output

After evaluation, you'll find:

- `results/figures/` - ROC curves, PR curves, metric comparisons, confusion matrices, radar charts
- `results/tables/` - CSV files with all performance metrics
- `results/evaluation_summary_report.md` - Text summary of the results

## Notes on Reproducibility

The training logs everything to `logs/` with timestamps. Each model directory contains the trained model plus all the preprocessing objects (scalers, feature selectors) and the feature names, so you can reload and make predictions consistently.

If TabPFN is giving you trouble on CPU, that's expected - it's designed for GPU. The code falls back to CPU but it's painfully slow. For the tree methods, CPU is fine, just slower.

## License

This is research code. Use it, modify it, but if you break it, you get to keep both pieces.

## Citation

If this is useful for your work, cite it as:

```
Foundation Models in Bioinformatics: Protein Antigenicity Prediction
GitHub: https://github.com/sebamiles/FoundationModelsInBioinformatics
```

## Contact

Issues on GitHub work, or reach out through [smilesinformatics.com](https://smilesinformatics.com)
