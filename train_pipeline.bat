@echo off
echo ========================================
echo IApred Training Pipeline
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

REM Check if requirements are installed
echo Checking dependencies...
python -c "import numpy, pandas, sklearn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)
echo Dependencies OK.
echo.

REM Check if models directory exists
if not exist "models" (
    echo Creating models directory...
    mkdir models
)

REM Run training
echo ========================================
echo Starting model training...
echo ========================================
echo.
python scripts/train_models.py

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo Training completed with errors!
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training completed successfully!
echo ========================================
echo Models saved to: models\
echo Results saved to: results\
echo.

pause
