#!/bin/bash

echo "Starting Image Colorization App..."
echo

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        echo "Please ensure Python 3 is installed and accessible"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

# Download model files
echo "Downloading model files..."
python download_models.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to download model files"
    exit 1
fi

# Set Flask app and run
echo "Starting Flask application..."
export FLASK_APP=app.py
python app.py



