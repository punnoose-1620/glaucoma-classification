#!/usr/bin/env python3
"""
Comprehensive runner for the Glaucoma Classification Project
This script provides an interactive way to run different parts of the project
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_environment():
    """Check if the environment is properly set up"""
    print("Checking environment...")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment may not be activated")
        print("   Run: .\\venv\\Scripts\\Activate.ps1")
    
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} is available")
    except ImportError:
        print("✗ TensorFlow is not available")
        return False
    
    # Check if dataset exists
    if os.path.exists("synthetic_glaucoma_data.h5"):
        print("✓ Dataset file found: synthetic_glaucoma_data.h5")
        return True
    else:
        print("✗ Dataset file not found: synthetic_glaucoma_data.h5")
        return False

def show_menu():
    """Show the main menu"""
    print("\n" + "="*60)
    print("GLAUCOMA CLASSIFICATION PROJECT RUNNER")
    print("="*60)
    print()
    print("Available options:")
    print("1. Run demo (no dataset required)")
    print("2. Download dataset from Kaggle")
    print("3. Run EfficientNet training")
    print("4. Run original CNN training")
    print("5. Run Vision Transformer training")
    print("6. Test existing model")
    print("7. Show project information")
    print("8. Exit")
    print()
    return input("Enter your choice (1-8): ")

def run_demo():
    """Run the demo script"""
    print("\nRunning demo...")
    subprocess.run([sys.executable, "demo_run.py"])

def download_dataset():
    """Run the dataset download script"""
    print("\nSetting up dataset download...")
    subprocess.run([sys.executable, "download_dataset.py"])

def run_efficientnet_training():
    """Run the EfficientNet training"""
    if not os.path.exists("synthetic_glaucoma_data.h5"):
        print("✗ Dataset not found. Please download the dataset first.")
        return
    
    print("\nRunning EfficientNet training...")
    print("This will take several hours and requires significant memory.")
    confirm = input("Continue? (y/n): ")
    if confirm.lower() == 'y':
        subprocess.run([sys.executable, "efficientnet_glaucoma_training.py"])

def run_cnn_training():
    """Run the original CNN training"""
    if not os.path.exists("synthetic_glaucoma_data.h5"):
        print("✗ Dataset not found. Please download the dataset first.")
        return
    
    print("\nRunning original CNN training...")
    print("This will take several hours.")
    confirm = input("Continue? (y/n): ")
    if confirm.lower() == 'y':
        subprocess.run([sys.executable, "glaucoma_classification.py"])

def run_vit_training():
    """Run the Vision Transformer training"""
    if not os.path.exists("synthetic_glaucoma_data.h5"):
        print("✗ Dataset not found. Please download the dataset first.")
        return
    
    print("\nRunning Vision Transformer training...")
    print("This will take several hours and requires significant memory.")
    confirm = input("Continue? (y/n): ")
    if confirm.lower() == 'y':
        subprocess.run([sys.executable, "model_vit_version.py"])

def test_model():
    """Test an existing model"""
    if not os.path.exists("synthetic_glaucoma_data.h5"):
        print("✗ Dataset not found. Please download the dataset first.")
        return
    
    print("\nRunning model testing...")
    subprocess.run([sys.executable, "test.py"])

def show_project_info():
    """Show detailed project information"""
    print("\n" + "="*60)
    print("PROJECT INFORMATION")
    print("="*60)
    print()
    print("Project: Glaucoma Classification using Deep Learning")
    print("Dataset: Synthetic OCT Glaucoma Dataset (3000 images)")
    print("Classes: 5 glaucoma types")
    print("Features: 6 different OCT imaging modalities")
    print()
    print("Model Architectures:")
    print("- EfficientNetB0 with transfer learning")
    print("- Custom multi-input CNN")
    print("- Vision Transformer (ViT)")
    print()
    print("Training Features:")
    print("- 5-fold cross-validation")
    print("- Advanced data augmentation")
    print("- Memory-efficient training")
    print("- Comprehensive logging")
    print("- Early stopping and learning rate scheduling")
    print()
    print("Performance (from README):")
    print("- EfficientNet: 20.2% accuracy")
    print("- CNN: 19.2% accuracy")
    print("- Both models show similar limitations due to dataset constraints")
    print()
    print("Files and Directories:")
    print("- models/: Saved model files")
    print("- logs/: Training logs and results")
    print("- visualizations/: Generated plots and figures")
    print("="*60)

def main():
    """Main function"""
    print("Welcome to the Glaucoma Classification Project!")
    
    # Check environment
    dataset_available = check_environment()
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            download_dataset()
        elif choice == '3':
            run_efficientnet_training()
        elif choice == '4':
            run_cnn_training()
        elif choice == '5':
            run_vit_training()
        elif choice == '6':
            test_model()
        elif choice == '7':
            show_project_info()
        elif choice == '8':
            print("\nThank you for using the Glaucoma Classification Project!")
            break
        else:
            print("Invalid choice. Please enter a number between 1-8.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
