#!/usr/bin/env python3
"""
Download IAPred SVM model from GitHub repository.

This script downloads the pre-trained IAPred SVM model from:
https://github.com/sebamiles/IAPred/tree/main/models

The model includes:
- IApred_SVM.joblib: The main SVM model
- IApred_all_feature_names.joblib: Feature names used by the model
- IApred_feature_mask.joblib: Feature mask configuration
- IApred_feature_selector.joblib: Feature selector
- IApred_scaler.joblib: Feature scaler
- IApred_variance_selector.joblib: Variance selector
"""

import os
import sys
import logging
from pathlib import Path
from urllib.request import urlretrieve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GitHub raw content URLs for the IAPred models
# Using raw.githubusercontent.com for direct downloads
IAPRED_MODEL_URLS = {
    'IApred_SVM.joblib': 'https://raw.githubusercontent.com/sebamiles/IAPred/main/models/IApred_SVM.joblib',
    'IApred_all_feature_names.joblib': 'https://raw.githubusercontent.com/sebamiles/IAPred/main/models/IApred_all_feature_names.joblib',
    'IApred_feature_mask.joblib': 'https://raw.githubusercontent.com/sebamiles/IAPred/main/models/IApred_feature_mask.joblib',
    'IApred_feature_selector.joblib': 'https://raw.githubusercontent.com/sebamiles/IAPred/main/models/IApred_feature_selector.joblib',
    'IApred_scaler.joblib': 'https://raw.githubusercontent.com/sebamiles/IAPred/main/models/IApred_scaler.joblib',
    'IApred_variance_selector.joblib': 'https://raw.githubusercontent.com/sebamiles/IAPred/main/models/IApred_variance_selector.joblib',
}

# Expected MD5 checksums for verification (optional - can be added later)
EXPECTED_CHECKSUMS = {}


def download_file(url, destination, chunk_size=8192):
    """Download a file from URL to destination."""
    try:
        logger.info(f"Downloading {os.path.basename(destination)}...")
        urlretrieve(url, destination)
        file_size = os.path.getsize(destination) / (1024 * 1024)  # Size in MB
        logger.info(f"  Downloaded: {destination} ({file_size:.2f} MB)")
        return True
    except Exception as e:
        logger.error(f"  Failed to download {os.path.basename(destination)}: {e}")
        return False


def download_iapred_model(output_dir=None):
    """Download all IAPred model files."""
    if output_dir is None:
        # Default to models directory
        output_dir = Path(__file__).parent.parent / 'models' / 'IApred_SVM'
    else:
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Downloading IAPred SVM Model")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")

    downloaded = 0
    failed = 0

    for filename, url in IAPRED_MODEL_URLS.items():
        destination = output_dir / filename
        if destination.exists():
            logger.info(f"  Skipping {filename} (already exists)")
            downloaded += 1
            continue

        if download_file(url, destination):
            downloaded += 1
        else:
            failed += 1

    logger.info("=" * 80)
    logger.info(f"Download complete: {downloaded} succeeded, {failed} failed")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 80)

    if failed > 0:
        logger.warning("Some files failed to download. Please check your internet connection.")
        return False

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download IAPred SVM model from GitHub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default models/IApred_SVM directory
  python scripts/download_iapred_model.py

  # Download to custom directory
  python scripts/download_iapred_model.py -o models/IApred_SVM
        """
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for model files (default: models/IApred_SVM)'
    )

    args = parser.parse_args()

    success = download_iapred_model(args.output_dir)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
