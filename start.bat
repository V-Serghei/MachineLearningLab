@echo off
setlocal

echo [MachineLearningLab] Checking prerequisites...

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    exit /b 1
)

REM Create virtual environment if it does not exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install --quiet pandas numpy matplotlib seaborn scikit-learn orange3 scipy

REM Extract credit card dataset if missing
if not exist "data\creditcard.csv" (
    echo Extracting data\creditcard.tar.xz...
    cd data
    tar -xvf creditcard.tar.xz
    cd ..
)

REM Smoke-test imports
echo Running import smoke test...
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, Orange, scipy; print('All imports OK')"
if errorlevel 1 (
    echo ERROR: Import smoke test failed. Check your environment.
    exit /b 1
)

echo.
echo [MachineLearningLab] Environment ready.
echo Example: python LB2\AdaBoost.py
