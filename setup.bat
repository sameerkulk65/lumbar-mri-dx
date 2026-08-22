@echo off
cd /d "%~dp0"
echo ============================================
echo  Lumbar MRI Diagnostic AI - Setup
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on this PC.
    echo Install Python 3.11+ from https://www.python.org/downloads/ ^(check "Add to PATH"^) and rerun this script.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo Virtual environment already exists, skipping creation.
)

echo.
echo Installing dependencies - this can take several minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements-demo.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency install failed. Check the messages above.
    pause
    exit /b 1
)

echo.
echo Preparing the trained model for inference...
if not exist "outputs\checkpoints\last_state_dict.pt" (
    echo [ERROR] outputs\checkpoints\last_state_dict.pt is missing from this checkout.
    echo Make sure you cloned the full repository ^(this file is required^).
    pause
    exit /b 1
)
".venv\Scripts\python.exe" export_trained_model.py outputs\checkpoints\last_state_dict.pt
if errorlevel 1 (
    echo.
    echo [ERROR] Model export failed. Check the messages above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Setup complete. Double-click run_demo.bat to launch.
echo ============================================
pause
