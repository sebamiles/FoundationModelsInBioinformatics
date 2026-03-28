# IApred-PFN Predictor

A command-line tool for predicting protein antigenicity using the TabPFN machine learning model trained on IApred-PFN dataset.

## Overview

This predictor uses the TabPFN model trained with all features to classify proteins as antigens or non-antigens. The model provides a score between 0 and 1, where higher scores indicate higher antigenicity potential.

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required dependencies from the local file:
   ```bash
   pip install -r requirements.txt
   ```
3. The predictor ships with pre-trained model files in the `models/` directory

## Usage

### Basic Usage

#### Single Sequence Prediction
```bash
python predictor.py -s "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
```

#### FASTA File Prediction
```bash
python predictor.py -f input_sequences.fasta
```

### Output Options

#### Save Results to CSV
```bash
python predictor.py -f input.fasta --csv results.csv
```

The CSV file will contain columns:
- **Header**: Original FASTA header (or "Input_Sequence" for single sequences)
- **Sequence**: Amino acid sequence
- **Score**: Antigenicity score (0-1, higher = more antigenic)
- **Category**: Classification ("antigen" or "non-antigen")

#### Save Results to FASTA
```bash
python predictor.py -f input.fasta --fasta-out output.fasta
```

The output FASTA file will have headers modified to include scores and categories:
```
>original_header|score=0.875|category=antigen
SEQUENCE...
```

#### Combined Output
```bash
python predictor.py -f input.fasta --csv results.csv --fasta-out output.fasta
```

### Advanced Options

#### Hide Scores in Terminal Output
```bash
python predictor.py -f input.fasta --no-scores
```

#### Custom Model Directory
```bash
python predictor.py -f input.fasta --model-dir /path/to/models
```

## Input Format

### Single Sequences
- Amino acid sequences containing standard 20 amino acids
- Non-standard characters are automatically filtered out
- Case insensitive

### FASTA Files
- Standard FASTA format with sequence headers
- Sequences are cleaned to contain only standard amino acids
- Empty sequences after cleaning are skipped

## Output Interpretation

### Score
- **Range**: 0.0 to 1.0
- **Higher values**: More likely to be antigenic (probability of antigen class)
- **Threshold**: 0.5 (scores ≥ 0.5 classified as "antigen")

### Category
- **antigen**: Score ≥ 0.5
- **non-antigen**: Score < 0.5

## Model Details

- **Algorithm**: TabPFN (Tabular Prior-Data Fitted Network)
- **Features**: All available features (sequence composition, physicochemical properties, motifs, etc.)
- **Training Data**: Balanced dataset of antigens and non-antigens from various pathogens
- **Performance**: High accuracy on external validation datasets

## File Structure

```
Predictor/
├── predictor.py          # Main prediction script
├── models/              # Pre-trained model files
│   ├── IApred_TabPFN.joblib
│   ├── IApred_variance_selector.joblib
│   └── IApred_all_feature_names.joblib
├── antigens/             # Training antigen FASTA files
├── non-antigens/         # Training non-antigen FASTA files
├── data/                 # External evaluation datasets
├── protein_motifs.txt    # (Optional) motif patterns for feature extraction
├── requirements.txt      # Local dependency list
├── functions_for_training.py  # Feature extraction functions
├── data_loader.py        # Data loading utilities
└── README.md             # This file
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`

2. **CUDA Errors**: The model will automatically use CPU if CUDA is not available

3. **Memory Issues**: For large FASTA files, consider processing in batches

4. **Invalid Sequences**: Sequences with no valid amino acids after cleaning will be skipped

### Getting Help

If you encounter issues:
1. Check that all model files are present in the `models/` directory
2. Verify Python version (3.7+ recommended)
3. Ensure all dependencies are properly installed
4. Check input file formats

## Citation

If you use this predictor in your research, please cite the original IApred-PFN paper:

```
[Add appropriate citation here]
```

## License

[Add license information if applicable]
