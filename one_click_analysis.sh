#!/bin/bash
# One-click analysis script for IApred publication
# This script trains all models (including downloading IApred SVM), runs evaluation, and computes statistics

echo "================================================"
echo "IApred Publication - One-Click Analysis"
echo "================================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

echo ""
echo "Step 1: Training all models (including downloading IApred SVM)..."
echo ""

# Train all models (this will also download IApred SVM)
python scripts/train_models.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Model training failed"
    exit 1
fi

echo ""
echo "================================================"
echo "Step 2: Running Evaluation on External Data"
echo "================================================"
echo ""

# Run evaluation
python scripts/evaluate_models.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Evaluation failed"
    exit 1
fi

echo ""
echo "================================================"
echo "Step 3: Computing Statistical Tests"
echo "================================================"
echo ""

# Compute statistics
python scripts/compute_statistics.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Statistical computation failed"
    exit 1
fi

echo ""
echo "================================================"
echo "Analysis Complete!"
echo "Results saved to: results/"
echo "  - Performance metrics: results/tables/table_performance_metrics.csv"
echo "  - Statistical tests: results/tables/table_statistical_tests.csv"
echo "  - Figures: results/figures/"
echo "================================================"
echo ""
