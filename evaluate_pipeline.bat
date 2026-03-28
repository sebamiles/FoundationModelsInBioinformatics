@echo off
echo ========================================
echo IApred Evaluation Pipeline
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if models exist
if not exist "models" (
    echo ERROR: Models directory not found. Please run training first.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting model evaluation...
echo ========================================
echo.
python scripts/evaluate_models.py

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo Evaluation completed with errors!
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo Evaluation completed successfully!
echo ========================================
echo Results saved to: results\
echo Figures saved to: results\figures\
echo Tables saved to: results\tables\
echo.

pause
