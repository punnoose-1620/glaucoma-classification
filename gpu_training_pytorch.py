#!/usr/bin/env python3
"""
GPU Training Script using PyTorch with CUDA
This script uses PyTorch to leverage your NVIDIA RTX 4060 GPU
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import logging
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import psutil
import gc

# Check if CUDA is available
def check_cuda():
    """Check CUDA availability and configure GPU"""
    print("="*60)
    print("NVIDIA GPU CONFIGURATION WITH PYTORCH")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Set default device to GPU
        device = torch.device('cuda:0')
        print(f"✓ Using device: {device}")
        
        # Set memory growth
        torch.cuda.empty_cache()
        print("✓ GPU memory cleared")
        
        return device
    else:
        print("✗ CUDA is not available")
        print("  Using CPU instead")
        return torch.device('cpu')

def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"gpu_training_pytorch_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

class GlaucomaDataset(Dataset):
    """Custom dataset for glaucoma classification"""
    
    def __init__(self, h5_file, indices=None):
        self.h5_file = h5_file
        self.indices = indices
        
        with h5py.File(h5_file, 'r') as f:
            if indices is None:
                self.num_samples = len(f['labels'])
                self.indices = np.arange(self.num_samples)
            else:
                self.num_samples = len(indices)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # Get the actual index
            actual_idx = self.indices[idx]
            
            # Load labels
            labels = f['labels'][actual_idx]
            
            # Load all feature types and combine them
            feature_types = ['rnfl_input', 'cup_disc_input', 'rim_input', 
                           'juxta_input', 'sheath_input', 'macular_input']
            
            combined_features = []
            for feature_type in feature_types:
                if feature_type in f['features']:
                    feature_data = f['features'][feature_type][actual_idx]
                    combined_features.append(feature_data)
            
            # Combine all features along the channel dimension
            if combined_features:
                # Stack features along the first dimension (channels)
                combined_inputs = np.stack(combined_features, axis=0)
            else:
                # Fallback: use first available feature
                first_feature = list(f['features'].keys())[0]
                combined_inputs = f['features'][first_feature][actual_idx]
                # Add channel dimension
                combined_inputs = np.expand_dims(combined_inputs, axis=0)
        
        # Normalize to [0, 1] range
        combined_inputs = np.clip(combined_inputs, 0, 255) / 255.0
        
        # Convert to PyTorch tensors
        inputs = torch.FloatTensor(combined_inputs)
        labels = torch.FloatTensor(labels)
        
        return inputs, labels

class GlaucomaCNN(nn.Module):
    """CNN model for glaucoma classification"""
    
    def __init__(self, num_classes=5):
        super(GlaucomaCNN, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # Clip inputs to avoid log(0)
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)
        
        # Calculate cross entropy
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal loss
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Convert one-hot labels to class indices
        labels_indices = torch.argmax(labels, dim=1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_indices)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels_indices.size(0)
        correct += (predicted == labels_indices).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Convert one-hot labels to class indices
            labels_indices = torch.argmax(labels, dim=1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels_indices)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_indices.size(0)
            correct += (predicted == labels_indices).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def train_fold(fold, train_dataset, val_dataset, device, logger):
    """Train a single fold"""
    logger.info(f"Starting training for Fold {fold}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Create model
    model = GlaucomaCNN(num_classes=5).to(device)
    
    # Create loss function and optimizer
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10
    
    # Training loop
    for epoch in range(50):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/50 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'models/gpu_pytorch_model_fold_{fold}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(f'models/gpu_pytorch_model_fold_{fold}_best.pth'))
    
    # Final evaluation
    final_val_loss, final_val_acc = validate_epoch(model, val_loader, criterion, device)
    logger.info(f"Fold {fold} - Final Validation Accuracy: {final_val_acc:.2f}%")
    
    return model, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }, final_val_acc

def create_visualizations(history, fold, logger):
    """Create training visualizations"""
    logger.info(f"Creating visualizations for Fold {fold}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training history
    axes[0].plot(history['train_accs'], label='Training Accuracy')
    axes[0].plot(history['val_accs'], label='Validation Accuracy')
    axes[0].set_title(f'Model Accuracy - Fold {fold}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_losses'], label='Training Loss')
    axes[1].plot(history['val_losses'], label='Validation Loss')
    axes[1].set_title(f'Model Loss - Fold {fold}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/gpu_pytorch_training_history_fold_{fold}.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main training function"""
    print("="*60)
    print("GPU GLAUCOMA CLASSIFICATION TRAINING WITH PYTORCH")
    print("="*60)
    
    # Check CUDA and setup device
    device = check_cuda()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting GPU training with PyTorch")
    
    # Check if dataset exists
    if not os.path.exists("synthetic_glaucoma_data.h5"):
        logger.error("Dataset not found: synthetic_glaucoma_data.h5")
        print("Please download the dataset first using download_dataset.py")
        return
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Load dataset info
    with h5py.File("synthetic_glaucoma_data.h5", 'r') as f:
        num_samples = len(f['labels'])
        logger.info(f"Dataset loaded: {num_samples} samples")
        logger.info(f"Feature types: {list(f['features'].keys())}")
    
    # Setup cross-validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Load labels for stratification
    with h5py.File("synthetic_glaucoma_data.h5", 'r') as f:
        labels_one_hot = f['labels'][:]
        # Convert one-hot labels to integer labels for stratification
        labels = np.argmax(labels_one_hot, axis=1)
        logger.info(f"Label distribution: {np.bincount(labels)}")
    
    # Training results
    fold_accuracies = []
    
    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(num_samples), labels), 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"TRAINING FOLD {fold}/{n_folds}")
        logger.info(f"{'='*40}")
        
        # Create datasets
        train_dataset = GlaucomaDataset("synthetic_glaucoma_data.h5", train_idx)
        val_dataset = GlaucomaDataset("synthetic_glaucoma_data.h5", val_idx)
        
        # Train the fold
        model, history, val_accuracy = train_fold(fold, train_dataset, val_dataset, device, logger)
        
        # Create visualizations
        create_visualizations(history, fold, logger)
        
        # Store results
        fold_accuracies.append(val_accuracy)
        
        # Clear memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Log memory usage
        memory_usage = psutil.virtual_memory().percent
        logger.info(f"Memory usage: {memory_usage:.1f}%")
    
    # Final results
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Average accuracy across {n_folds} folds: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")
    logger.info(f"Individual fold accuracies: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
    
    print(f"\nTraining completed! Average accuracy: {np.mean(fold_accuracies):.2f}%")
    print(f"Models saved in 'models/' directory")
    print(f"Visualizations saved in 'visualizations/' directory")

if __name__ == "__main__":
    main()
