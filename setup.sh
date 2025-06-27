#!/bin/bash

# Remove existing venv if it exists
rm -rf venv

# Create virtual environment with Python 3.10
/opt/homebrew/opt/python@3.10/bin/python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p visualizations logs models

# Set up Kaggle credentials
echo "Please make sure you have your Kaggle API credentials (kaggle.json) in ~/.kaggle/"
echo "You can download it from https://www.kaggle.com/settings/account"
echo "After placing the file, run: chmod 600 ~/.kaggle/kaggle.json"

echo "Setup complete! Don't forget to activate the virtual environment with: source venv/bin/activate" 