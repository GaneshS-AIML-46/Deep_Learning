@echo off
echo Starting Image Colorization App...
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        echo Please ensure Python is installed and accessible
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

REM Download model files
echo Downloading model files...
python download_models.py
if errorlevel 1 (
    echo Error: Failed to download model files
    pause
    exit /b 1
)

REM Set Flask app and run
echo Starting Flask application...
set FLASK_APP=app.py
python app.py

pause



