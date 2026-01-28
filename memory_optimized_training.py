#!/usr/bin/env python3
"""
Memory-Optimized GPU Training Script
Limits RAM usage to 10GB while maximizing GPU utilization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import h5py
import logging
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import psutil
import gc
import warnings
import threading
import time
from collections import deque
warnings.filterwarnings('ignore')

# Memory monitoring and limiting
class MemoryManager:
    """Manages memory usage to stay within 10GB limit"""
    
    def __init__(self, max_ram_gb=10.0):
        self.max_ram_gb = max_ram_gb
        self.max_ram_bytes = max_ram_gb * 1024**3
        self.memory_history = deque(maxlen=100)
        self.monitoring = False
        
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        
    def _monitor_memory(self):
        """Monitor memory usage and log warnings"""
        while self.monitoring:
            current_memory = psutil.virtual_memory()
            self.memory_history.append(current_memory.percent)
            
            if current_memory.used > self.max_ram_bytes:
                print(f"⚠️  WARNING: Memory usage {current_memory.used / 1024**3:.1f}GB exceeds limit of {self.max_ram_gb}GB")
                self._force_garbage_collection()
            
            time.sleep(2)  # Check every 2 seconds
    
    def _force_garbage_collection(self):
        """Force garbage collection to free memory"""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    def get_memory_info(self):
        """Get current memory information"""
        memory = psutil.virtual_memory()
        gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        
        return {
            'ram_used_gb': memory.used / 1024**3,
            'ram_percent': memory.percent,
            'gpu_used_gb': gpu_memory,
            'gpu_total_gb': gpu_total,
            'gpu_percent': (gpu_memory / gpu_total * 100) if gpu_total > 0 else 0
        }

# Optimized GPU Configuration
def setup_optimized_gpu():
    """Optimized GPU configuration for maximum GPU utilization and minimal CPU usage"""
    print("="*60)
    print("MEMORY-OPTIMIZED GPU CONFIGURATION")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        
        # Get GPU info
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Set device
        device = torch.device('cuda:0')
        
        # Advanced GPU optimizations for maximum utilization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Set memory fraction to use more GPU memory (95% instead of 90%)
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        print(f"✓ Using device: {device}")
        print("✓ Optimized GPU settings:")
        print("  - cuDNN benchmark mode")
        print("  - TensorFloat-32 enabled")
        print("  - Memory fraction set to 95%")
        print("  - RAM limit: 10GB")
        
        return device
    else:
        print("✗ CUDA is not available")
        return torch.device('cpu')

def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"memory_optimized_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

class OptimizedGlaucomaDataset(Dataset):
    """Memory-optimized dataset with efficient data loading"""
    
    def __init__(self, h5_file, indices=None, is_training=True, memory_manager=None):
        self.h5_file = h5_file
        self.indices = indices
        self.is_training = is_training
        self.memory_manager = memory_manager
        
        with h5py.File(h5_file, 'r') as f:
            if indices is None:
                self.num_samples = len(f['labels'])
                self.indices = np.arange(self.num_samples)
            else:
                self.num_samples = len(indices)
        
        # Simplified transforms to reduce CPU usage
        if self.is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),  # Reduced from 15
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Check memory before loading data
        if self.memory_manager:
            mem_info = self.memory_manager.get_memory_info()
            if mem_info['ram_used_gb'] > 9.5:  # Force GC if approaching limit
                self.memory_manager._force_garbage_collection()
        
        with h5py.File(self.h5_file, 'r') as f:
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
            
            # Combine features
            if combined_features:
                combined_inputs = np.stack(combined_features, axis=0)
            else:
                first_feature = list(f['features'].keys())[0]
                combined_inputs = f['features'][first_feature][actual_idx]
                combined_inputs = np.expand_dims(combined_inputs, axis=0)
        
        # Normalize to [0, 1] range
        combined_inputs = np.clip(combined_inputs, 0, 255) / 255.0
        
        # Convert 6-channel to 3-channel RGB by averaging pairs of channels
        if combined_inputs.shape[0] == 6:
            combined_inputs = combined_inputs.reshape(3, 2, 224, 224).mean(axis=1)
        elif combined_inputs.shape[0] == 1:
            combined_inputs = np.repeat(combined_inputs, 3, axis=0)
        
        # Ensure we have 3 channels
        if combined_inputs.shape[0] != 3:
            if combined_inputs.shape[0] > 3:
                combined_inputs = combined_inputs[:3]
            else:
                combined_inputs = np.repeat(combined_inputs[:1], 3, axis=0)
        
        # Convert to PIL Image for augmentation
        combined_inputs = (combined_inputs * 255).astype(np.uint8)
        combined_inputs = combined_inputs.transpose(1, 2, 0)  # (H, W, C)
        
        # Apply transforms
        combined_inputs = self.transform(combined_inputs)
        
        # Convert to PyTorch tensors
        labels = torch.FloatTensor(labels)
        
        return combined_inputs, labels

class EfficientGlaucomaCNN(nn.Module):
    """Efficient CNN optimized for GPU memory usage"""
    
    def __init__(self, num_classes=5):
        super(EfficientGlaucomaCNN, self).__init__()
        
        # Efficient architecture with fewer parameters
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # First block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Efficient classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
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
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal loss
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()

def train_epoch_optimized(model, dataloader, criterion, optimizer, device, memory_manager=None):
    """Memory-optimized training for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Check memory before processing batch
        if memory_manager:
            mem_info = memory_manager.get_memory_info()
            if mem_info['ram_used_gb'] > 9.5:
                memory_manager._force_garbage_collection()
        
        inputs = inputs.to(device, non_blocking=True)  # Use non_blocking for faster transfer
        labels = labels.to(device, non_blocking=True)
        
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
        
        # Clear intermediate variables to save memory
        del outputs, loss
        if batch_idx % 10 == 0:  # Periodic memory cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch_optimized(model, dataloader, criterion, device, memory_manager=None):
    """Memory-optimized validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Check memory before processing batch
            if memory_manager:
                mem_info = memory_manager.get_memory_info()
                if mem_info['ram_used_gb'] > 9.5:
                    memory_manager._force_garbage_collection()
            
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Convert one-hot labels to class indices
            labels_indices = torch.argmax(labels, dim=1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels_indices)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_indices.size(0)
            correct += (predicted == labels_indices).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels_indices.cpu().numpy())
            
            # Clear intermediate variables
            del outputs, loss
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_targets

def train_fold_optimized(fold, train_dataset, val_dataset, device, logger, memory_manager=None):
    """Memory-optimized training for a single fold"""
    logger.info(f"Starting memory-optimized training for Fold {fold}")
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Smaller batch size for memory efficiency
        shuffle=True, 
        num_workers=0,  # Single-threaded to avoid pickling issues
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0,  # Single-threaded to avoid pickling issues
        pin_memory=True
    )
    
    # Create efficient model
    model = EfficientGlaucomaCNN(num_classes=5).to(device)
    
    # Optimized loss function and optimizer
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10
    
    # Training loop
    for epoch in range(50):  # Reduced epochs for memory efficiency
        # Log memory usage
        if memory_manager:
            mem_info = memory_manager.get_memory_info()
            logger.info(f"Memory - RAM: {mem_info['ram_used_gb']:.1f}GB, GPU: {mem_info['gpu_used_gb']:.1f}GB")
        
        # Train
        train_loss, train_acc = train_epoch_optimized(
            model, train_loader, criterion, optimizer, device, memory_manager
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate_epoch_optimized(
            model, val_loader, criterion, device, memory_manager
        )
        
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'models/memory_optimized_model_fold_{fold}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Force memory cleanup after each epoch
        if memory_manager:
            memory_manager._force_garbage_collection()
    
    # Load best model
    checkpoint = torch.load(f'models/memory_optimized_model_fold_{fold}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    final_val_loss, final_val_acc, predictions, targets = validate_epoch_optimized(
        model, val_loader, criterion, device, memory_manager
    )
    logger.info(f"Fold {fold} - Final Validation Accuracy: {final_val_acc:.2f}%")
    
    return model, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'predictions': predictions,
        'targets': targets
    }, final_val_acc

def create_optimized_visualizations(history, fold, logger):
    """Create training visualizations"""
    logger.info(f"Creating visualizations for Fold {fold}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training history
    axes[0].plot(history['train_accs'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history['val_accs'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title(f'Model Accuracy - Fold {fold}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_losses'], label='Training Loss', linewidth=2)
    axes[1].plot(history['val_losses'], label='Validation Loss', linewidth=2)
    axes[1].set_title(f'Model Loss - Fold {fold}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/memory_optimized_training_history_fold_{fold}.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main memory-optimized training function"""
    print("="*60)
    print("MEMORY-OPTIMIZED GPU GLAUCOMA CLASSIFICATION TRAINING")
    print("="*60)
    
    # Initialize memory manager
    memory_manager = MemoryManager(max_ram_gb=10.0)
    memory_manager.start_monitoring()
    
    # Optimized GPU setup
    device = setup_optimized_gpu()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting memory-optimized GPU training")
    
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
        labels = np.argmax(labels_one_hot, axis=1)
        logger.info(f"Label distribution: {np.bincount(labels)}")
    
    # Training results
    fold_accuracies = []
    all_predictions = []
    all_targets = []
    
    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(num_samples), labels), 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"MEMORY-OPTIMIZED TRAINING FOLD {fold}/{n_folds}")
        logger.info(f"{'='*40}")
        
        # Create optimized datasets
        train_dataset = OptimizedGlaucomaDataset("synthetic_glaucoma_data.h5", train_idx, is_training=True, memory_manager=memory_manager)
        val_dataset = OptimizedGlaucomaDataset("synthetic_glaucoma_data.h5", val_idx, is_training=False, memory_manager=memory_manager)
        
        # Train the fold
        model, history, val_accuracy = train_fold_optimized(fold, train_dataset, val_dataset, device, logger, memory_manager)
        
        # Create visualizations
        create_optimized_visualizations(history, fold, logger)
        
        # Store results
        fold_accuracies.append(val_accuracy)
        all_predictions.extend(history['predictions'])
        all_targets.extend(history['targets'])
        
        # Clear memory
        del model, train_dataset, val_dataset
        memory_manager._force_garbage_collection()
        
        # Log memory usage
        mem_info = memory_manager.get_memory_info()
        logger.info(f"Memory usage: RAM {mem_info['ram_used_gb']:.1f}GB, GPU {mem_info['gpu_used_gb']:.1f}GB")
    
    # Stop memory monitoring
    memory_manager.stop_monitoring()
    
    # Final results
    logger.info(f"\n{'='*60}")
    logger.info("MEMORY-OPTIMIZED TRAINING COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Average accuracy across {n_folds} folds: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")
    logger.info(f"Individual fold accuracies: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
    
    # Calculate additional metrics
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Memory Optimized Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('visualizations/memory_optimized_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Classification report
    report = classification_report(all_targets, all_predictions, output_dict=True)
    logger.info(f"Classification Report:\n{classification_report(all_targets, all_predictions)}")
    
    print(f"\nMemory-optimized training completed!")
    print(f"Average accuracy: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")
    print(f"Best fold accuracy: {max(fold_accuracies):.2f}%")
    print(f"Models saved in 'models/' directory")
    print(f"Visualizations saved in 'visualizations/' directory")

if __name__ == "__main__":
    main()
