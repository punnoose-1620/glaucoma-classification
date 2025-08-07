#!/usr/bin/env python3
"""
Script to download the glaucoma dataset from Kaggle
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_kaggle_credentials():
    """Check if Kaggle credentials are set up"""
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_path.exists()

def setup_kaggle_instructions():
    """Print instructions for setting up Kaggle credentials"""
    print("\n" + "="*60)
    print("KAGGLE API SETUP REQUIRED")
    print("="*60)
    print("To download the dataset, you need to set up Kaggle API credentials:")
    print()
    print("1. Go to https://www.kaggle.com/settings/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download the kaggle.json file")
    print("5. Place the file in your user directory:")
    print(f"   {Path.home()}\\.kaggle\\kaggle.json")
    print()
    print("6. Set proper permissions (run in PowerShell as Administrator):")
    print(f'   icacls "{Path.home()}\\.kaggle\\kaggle.json" /inheritance:r /grant:r "{os.getenv("USERNAME")}:F"')
    print()
    print("After setting up credentials, run this script again.")
    print("="*60)

def download_dataset():
    """Download the dataset from Kaggle"""
    try:
        print("Downloading dataset from Kaggle...")
        import kaggle
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            'aleksanderprudnik/synthetic-oct-glaucoma-dataset',
            path='.',
            quiet=False,
            unzip=True
        )
        
        print("Dataset downloaded successfully!")
        print("Files downloaded:")
        for file in os.listdir('.'):
            if file.endswith('.h5'):
                print(f"  - {file}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False

def main():
    """Main function"""
    print("Glaucoma Classification Dataset Downloader")
    print("="*50)
    
    # Check if dataset already exists
    if os.path.exists("synthetic_glaucoma_data.h5"):
        print("Dataset already exists: synthetic_glaucoma_data.h5")
        return
    
    # Check Kaggle credentials
    if not check_kaggle_credentials():
        setup_kaggle_instructions()
        return
    
    # Try to download the dataset
    if download_dataset():
        print("\nDataset download completed successfully!")
        print("You can now run the training script:")
        print("  python efficientnet_glaucoma_training.py")
    else:
        print("\nFailed to download dataset. Please check your Kaggle credentials.")

if __name__ == "__main__":
    main()
