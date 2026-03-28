#!/bin/bash
# IApred Training Pipeline for Linux/Mac

echo "========================================"
echo "IApred Training Pipeline"
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

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import numpy, pandas, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
fi
echo "Dependencies OK."
echo ""

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
fi

# Run training
echo "========================================"
echo "Starting model training..."
echo "========================================"
echo ""
python3 scripts/train_models.py

if [ $? -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "Training completed with errors!"
    echo "========================================"
    exit 1
fi

echo ""
echo "========================================"
echo "Training completed successfully!"
echo "========================================"
echo "Models saved to: models/"
echo "Results saved to: results/"
echo ""

echo "Done!"
