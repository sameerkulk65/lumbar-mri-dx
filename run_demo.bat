@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Setup has not been run yet.
    echo Double-click setup.bat first, then try again.
    pause
    exit /b 1
)

echo Starting Lumbar MRI Diagnostic AI...
echo Your browser will open automatically at http://localhost:8501
echo Close this window to stop the app.
echo.
".venv\Scripts\streamlit.exe" run frontend\app.py
pause
