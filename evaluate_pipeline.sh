#!/bin/bash
# IApred Evaluation Pipeline for Linux/Mac

echo "========================================"
echo "IApred Evaluation Pipeline"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed or not in PATH"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check if models exist
if [ ! -d "models" ]; then
    echo "ERROR: Models directory not found. Please run training first."
    exit 1
fi

echo ""
echo "========================================"
echo "Starting model evaluation..."
echo "========================================"
echo ""
python3 scripts/evaluate_models.py

if [ $? -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "Evaluation completed with errors!"
    echo "========================================"
    exit 1
fi

echo ""
echo "========================================"
echo "Evaluation completed successfully!"
echo "========================================"
echo "Results saved to: results/"
echo "Figures saved to: results/figures/"
echo "Tables saved to: results/tables/"
echo ""

echo "Done!"
