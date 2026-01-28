#!/usr/bin/env python3
"""
Advanced GPU Training Script with Sophisticated Models
Leveraging NVIDIA RTX 4060 with advanced techniques
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
warnings.filterwarnings('ignore')

# Advanced GPU Configuration
def setup_advanced_gpu():
    """Advanced GPU configuration for maximum performance"""
    print("="*60)
    print("ADVANCED NVIDIA GPU CONFIGURATION")
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
        
        # Advanced GPU optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat-32 for faster training
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Set memory fraction (use 90% of GPU memory)
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        print(f"✓ Using device: {device}")
        print("✓ Advanced GPU optimizations enabled")
        print("  - cuDNN benchmark mode")
        print("  - TensorFloat-32 enabled")
        print("  - Memory fraction set to 90%")
        
        return device
    else:
        print("✗ CUDA is not available")
        return torch.device('cpu')

def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"advanced_gpu_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

class AdvancedGlaucomaDataset(Dataset):
    """Advanced dataset with sophisticated augmentation"""
    
    def __init__(self, h5_file, indices=None, is_training=True):
        self.h5_file = h5_file
        self.indices = indices
        self.is_training = is_training
        
        with h5py.File(h5_file, 'r') as f:
            if indices is None:
                self.num_samples = len(f['labels'])
                self.indices = np.arange(self.num_samples)
            else:
                self.num_samples = len(indices)
        
        # Advanced data augmentation for training
        if self.is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
            # Reshape to (3, 224, 224) by averaging pairs of channels
            combined_inputs = combined_inputs.reshape(3, 2, 224, 224).mean(axis=1)
        elif combined_inputs.shape[0] == 1:
            # If only 1 channel, repeat it 3 times
            combined_inputs = np.repeat(combined_inputs, 3, axis=0)
        
        # Ensure we have 3 channels
        if combined_inputs.shape[0] != 3:
            # If still not 3 channels, take the first 3 or repeat the first channel
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

class ResidualBlock(nn.Module):
    """Residual block for ResNet-style architecture"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionModule(nn.Module):
    """Attention mechanism for better feature selection"""
    
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class AdvancedGlaucomaCNN(nn.Module):
    """Advanced CNN with ResNet-style architecture and attention"""
    
    def __init__(self, num_classes=5):
        super(AdvancedGlaucomaCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks with attention
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Attention modules
        self.attention1 = AttentionModule(64)
        self.attention2 = AttentionModule(128)
        self.attention3 = AttentionModule(256)
        self.attention4 = AttentionModule(512)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Advanced classifier with multiple heads
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        x = self.layer4(x)
        x = self.attention4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

class FocalLoss(nn.Module):
    """Advanced Focal Loss with label smoothing"""
    
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # Label smoothing
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            targets = F.one_hot(targets, num_classes).float()
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        # Clip inputs to avoid log(0)
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal loss
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()

def train_epoch_advanced(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Advanced training for one epoch with gradient accumulation"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Gradient accumulation steps
    accumulation_steps = 4
    optimizer.zero_grad()
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Convert one-hot labels to class indices
        labels_indices = torch.argmax(labels, dim=1)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels_indices)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        total += labels_indices.size(0)
        correct += (predicted == labels_indices).sum().item()
    
    # Update learning rate
    if scheduler is not None:
        scheduler.step()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch_advanced(model, dataloader, criterion, device):
    """Advanced validation with ensemble predictions"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Convert one-hot labels to class indices
            labels_indices = torch.argmax(labels, dim=1)
            
            # Test time augmentation (TTA)
            outputs = model(inputs)
            
            # Add horizontal flip for TTA
            inputs_flipped = torch.flip(inputs, [3])
            outputs_flipped = model(inputs_flipped)
            
            # Average predictions
            outputs = (outputs + outputs_flipped) / 2
            
            loss = criterion(outputs, labels_indices)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_indices.size(0)
            correct += (predicted == labels_indices).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels_indices.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_targets

def train_fold_advanced(fold, train_dataset, val_dataset, device, logger):
    """Advanced training for a single fold"""
    logger.info(f"Starting advanced training for Fold {fold}")
    
    # Create data loaders with larger batch size
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create advanced model
    model = AdvancedGlaucomaCNN(num_classes=5).to(device)
    
    # Advanced loss function and optimizer
    criterion = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Advanced learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 15
    
    # Training loop with more epochs
    for epoch in range(100):
        # Train
        train_loss, train_acc = train_epoch_advanced(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate_epoch_advanced(
            model, val_loader, criterion, device
        )
        
        # Store history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/100 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping with more patience
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'models/advanced_gpu_model_fold_{fold}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    checkpoint = torch.load(f'models/advanced_gpu_model_fold_{fold}_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    final_val_loss, final_val_acc, predictions, targets = validate_epoch_advanced(
        model, val_loader, criterion, device
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

def create_advanced_visualizations(history, fold, logger):
    """Create advanced training visualizations"""
    logger.info(f"Creating advanced visualizations for Fold {fold}")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot training history
    axes[0, 0].plot(history['train_accs'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history['val_accs'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title(f'Model Accuracy - Fold {fold}', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['train_losses'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history['val_losses'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title(f'Model Loss - Fold {fold}', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate curve
    epochs = range(1, len(history['train_accs']) + 1)
    lr_values = [0.001 * (0.5 ** (epoch // 10)) for epoch in epochs]
    axes[1, 0].plot(epochs, lr_values, 'g-', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy improvement
    if len(history['val_accs']) > 1:
        improvement = [history['val_accs'][i] - history['val_accs'][i-1] for i in range(1, len(history['val_accs']))]
        axes[1, 1].plot(range(2, len(history['val_accs']) + 1), improvement, 'r-', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Validation Accuracy Improvement', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Improvement (%)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/advanced_gpu_training_history_fold_{fold}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main advanced training function"""
    print("="*60)
    print("ADVANCED GPU GLAUCOMA CLASSIFICATION TRAINING")
    print("="*60)
    
    # Advanced GPU setup
    device = setup_advanced_gpu()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting advanced GPU training")
    
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
        logger.info(f"ADVANCED TRAINING FOLD {fold}/{n_folds}")
        logger.info(f"{'='*40}")
        
        # Create advanced datasets
        train_dataset = AdvancedGlaucomaDataset("synthetic_glaucoma_data.h5", train_idx, is_training=True)
        val_dataset = AdvancedGlaucomaDataset("synthetic_glaucoma_data.h5", val_idx, is_training=False)
        
        # Train the fold
        model, history, val_accuracy = train_fold_advanced(fold, train_dataset, val_dataset, device, logger)
        
        # Create advanced visualizations
        create_advanced_visualizations(history, fold, logger)
        
        # Store results
        fold_accuracies.append(val_accuracy)
        all_predictions.extend(history['predictions'])
        all_targets.extend(history['targets'])
        
        # Clear memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        # Log memory usage
        memory_usage = psutil.virtual_memory().percent
        gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        logger.info(f"Memory usage: {memory_usage:.1f}% (GPU: {gpu_memory:.2f} GB)")
    
    # Final results
    logger.info(f"\n{'='*60}")
    logger.info("ADVANCED TRAINING COMPLETED")
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
    plt.title('Confusion Matrix - Advanced Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('visualizations/advanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Classification report
    report = classification_report(all_targets, all_predictions, output_dict=True)
    logger.info(f"Classification Report:\n{classification_report(all_targets, all_predictions)}")
    
    print(f"\nAdvanced training completed!")
    print(f"Average accuracy: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")
    print(f"Best fold accuracy: {max(fold_accuracies):.2f}%")
    print(f"Models saved in 'models/' directory")
    print(f"Visualizations saved in 'visualizations/' directory")

if __name__ == "__main__":
    main()
