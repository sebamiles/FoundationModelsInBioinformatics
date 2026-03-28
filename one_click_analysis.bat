@echo off
REM One-click analysis script for IApred publication
REM This script trains all models (including downloading IApred SVM), runs evaluation, and computes statistics

echo ================================================================
echo IApred Publication - One-Click Analysis
echo ================================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Step 1: Training all models (including downloading IApred SVM)...
echo.

REM Train all models (this will also download IApred SVM)
python scripts/train_models.py

if %errorlevel% neq 0 (
    echo.
    echo Error: Model training failed
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Step 2: Running Evaluation on External Data
echo ================================================================
echo.

REM Run evaluation
python scripts/evaluate_models.py

if %errorlevel% neq 0 (
    echo.
    echo Error: Evaluation failed
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Step 3: Computing Statistical Tests
echo ================================================================
echo.

REM Compute statistics
python scripts/compute_statistics.py

if %errorlevel% neq 0 (
    echo.
    echo Error: Statistical computation failed
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Analysis Complete!
echo Results saved to: results/
echo   - Performance metrics: results/tables/table_performance_metrics.csv
echo   - Statistical tests: results/tables/table_statistical_tests.csv
echo   - Figures: results/figures/
echo ================================================================
echo.

pause
