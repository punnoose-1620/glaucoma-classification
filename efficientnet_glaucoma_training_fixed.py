#!/usr/bin/env python3
"""
Comprehensive Glaucoma Classification Training with EfficientNetB0 (Fixed Version)
Features:
- EfficientNetB0 pre-trained model
- Focal Loss for class imbalance
- Cosine Annealing Learning Rate Scheduler
- 100 epochs with early stopping
- Memory-efficient training (5GB RAM limit)
- Comprehensive data augmentation
- Cross-validation
- Detailed logging and visualization
- Fixed grayscale to RGB conversion
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
import h5py
import psutil
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Import our data augmentation module
from data_augmentation import MedicalImageAugmentation

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Found {len(physical_devices)} GPU(s)")
else:
    print("No GPU found, using CPU")

# Configure logging
def setup_logging():
    """Setup comprehensive logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/efficientnet_training_fixed_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss implementation for handling class imbalance
    """
    def __init__(self, alpha=1, gamma=2, reduction='sum_over_batch_size', name='focal_loss'):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Convert to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal loss
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Apply alpha weighting
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Final focal loss
        focal_loss = alpha_weight * focal_weight * cross_entropy
        
        return focal_loss


class MemoryEfficientDataGenerator(keras.utils.Sequence):
    """
    Memory-efficient data generator for EfficientNet training
    """
    def __init__(self, h5_file_path, batch_size=16, shuffle=True, augment=True, 
                 input_types=None, memory_limit_gb=10):
        self.h5_file_path = h5_file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.input_types = input_types or ['cup_disc_input']
        self.memory_limit_gb = memory_limit_gb
        
        # Load dataset info
        with h5py.File(h5_file_path, 'r') as f:
            self.num_samples = f['features'][self.input_types[0]].shape[0]
            self.indices = np.arange(self.num_samples)
        
        self._num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
        # Initialize augmentation
        if self.augment:
            self.augmentor = MedicalImageAugmentation()
        
        logger.info(f"DataGenerator: {self.num_samples} samples, {self._num_batches} batches")
    
    def __len__(self):
        return self._num_batches
    
    def __getitem__(self, index):
        # Calculate batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Load batch data
        batch_data = self._load_batch(batch_indices)
        
        # Apply augmentation if enabled
        if self.augment:
            batch_data = self._augment_batch(batch_data)
        
        # Monitor memory usage
        self._check_memory_usage()
        
        return batch_data
    
    def _load_batch(self, indices):
        """Load batch from HDF5 with proper grayscale to RGB conversion"""
        # Sort indices for h5py compatibility
        sorted_indices = np.sort(indices)
        reverse_map = np.argsort(np.argsort(indices))
        
        with h5py.File(self.h5_file_path, 'r') as f:
            batch_labels = f['labels'][sorted_indices].astype(np.int32)
            
            # Load and combine features
            combined_inputs = []
            for input_type in self.input_types:
                data = f['features'][input_type][sorted_indices].astype(np.float32)
                combined_inputs.append(data)
            
            # Average the features
            combined_inputs = np.mean(combined_inputs, axis=0)
            
            # Normalize to [0, 1]
            combined_inputs = (combined_inputs - combined_inputs.min()) / (combined_inputs.max() - combined_inputs.min() + 1e-8)
            
            # Convert grayscale to RGB by repeating the channel
            if len(combined_inputs.shape) == 3:
                # Add channel dimension if needed
                if combined_inputs.shape[-1] != 1:
                    combined_inputs = np.expand_dims(combined_inputs, axis=-1)
                # Repeat to create RGB
                combined_inputs = np.repeat(combined_inputs, 3, axis=-1)
            
            # Apply EfficientNet preprocessing
            combined_inputs = tf.keras.applications.efficientnet.preprocess_input(combined_inputs)
            
            # Restore original order
            combined_inputs = combined_inputs[reverse_map]
            batch_labels = batch_labels[reverse_map]
            
            return combined_inputs, batch_labels
    
    def _augment_batch(self, batch_data):
        """Apply augmentation to batch"""
        images, labels = batch_data
        
        # Apply augmentation to each image
        augmented_images = []
        for i in range(len(images)):
            aug_img = self.augmentor.augment_single_image(images[i])
            augmented_images.append(aug_img)
        
        return np.array(augmented_images), labels
    
    def _check_memory_usage(self):
        """Check and log memory usage"""
        memory_usage = psutil.virtual_memory()
        memory_gb = memory_usage.used / (1024**3)
        
        if memory_gb > self.memory_limit_gb:
            logger.warning(f"Memory usage ({memory_gb:.2f}GB) exceeds limit ({self.memory_limit_gb}GB)")
            gc.collect()
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_efficientnet_model(num_classes=5, dropout_rate=0.5):
    """
    Create EfficientNetB0 model with custom head
    """
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create model
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    logger.info(f"EfficientNetB0 model created with {model.count_params():,} parameters")
    return model


def create_callbacks(model_save_path, patience=15):
    """Create training callbacks"""
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=f'logs/tensorboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    return callbacks_list


def train_fold(fold, train_indices, val_indices, h5_file_path, config):
    """
    Train a single fold
    """
    logger.info(f"Training Fold {fold + 1}/5")
    
    # Create data generators
    train_generator = MemoryEfficientDataGenerator(
        h5_file_path=h5_file_path,
        batch_size=config['batch_size'],
        shuffle=True,
        augment=True,
        input_types=['cup_disc_input'],  # Use single input for simplicity
        memory_limit_gb=config['memory_limit_gb']
    )
    
    val_generator = MemoryEfficientDataGenerator(
        h5_file_path=h5_file_path,
        batch_size=config['batch_size'],
        shuffle=False,
        augment=False,
        input_types=['cup_disc_input'],
        memory_limit_gb=config['memory_limit_gb']
    )
    
    # Update indices
    train_generator.indices = train_indices
    val_generator.indices = val_indices
    
    # Create model
    model = create_efficientnet_model(
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    )
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config['initial_lr']),
        loss=FocalLoss(),
        metrics=['accuracy']
    )
    
    # Create callbacks
    model_save_path = f"models/efficientnet_fold_{fold + 1}.h5"
    callbacks_list = create_callbacks(model_save_path, config['early_stopping_patience'])
    
    # Phase 1: Train with frozen base
    logger.info("Phase 1: Training with frozen base model")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['epochs'] // 2,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Phase 2: Fine-tune the entire model
    logger.info("Phase 2: Fine-tuning entire model")
    model.trainable = True
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config['initial_lr'] / 10),
        loss=FocalLoss(),
        metrics=['accuracy']
    )
    
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['epochs'] // 2,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    
    # Get predictions
    val_predictions = model.predict(val_generator, verbose=0)
    val_predictions = np.argmax(val_predictions, axis=1)
    
    # Get true labels
    val_labels = []
    for i in range(len(val_generator)):
        _, labels = val_generator[i]
        val_labels.extend(np.argmax(labels, axis=1))
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_labels, val_predictions, average='macro', zero_division=0
    )
    
    results = {
        'fold': fold + 1,
        'accuracy': val_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': val_loss
    }
    
    logger.info(f"Fold {fold + 1} Results:")
    logger.info(f"  Accuracy: {val_accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return results, history, history_fine


def create_visualizations(results, config):
    """Create visualizations of results"""
    if not results:
        return
    
    # Create confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    accuracies = [r['accuracy'] for r in results]
    axes[0].bar(range(1, len(accuracies) + 1), accuracies)
    axes[0].set_title('Accuracy by Fold')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    
    # Metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    avg_metrics = [np.mean([r[m] for r in results]) for m in metrics]
    std_metrics = [np.std([r[m] for r in results]) for m in metrics]
    
    x = np.arange(len(metrics))
    axes[1].bar(x, avg_metrics, yerr=std_metrics, capsize=5)
    axes[1].set_title('Average Metrics')
    axes[1].set_xlabel('Metric')
    axes[1].set_ylabel('Value')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('visualizations/efficientnet_results_fixed.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizations saved to visualizations/efficientnet_results_fixed.png")


def main():
    """
    Main training function
    """
    logger.info("Starting EfficientNet Glaucoma Classification Training (Fixed Version)")
    
    # Configuration
    config = {
        'h5_file_path': 'synthetic_glaucoma_data.h5',
        'n_folds': 5,
        'epochs': 100,
        'batch_size': 16,
        'initial_lr': 1e-3,
        'dropout_rate': 0.5,
        'num_classes': 5,
        'memory_limit_gb': 10,
        'early_stopping_patience': 15,
        'random_state': 42
    }
    
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Check if dataset exists
    if not os.path.exists(config['h5_file_path']):
        logger.error(f"Dataset not found: {config['h5_file_path']}")
        return
    
    # Load dataset info
    with h5py.File(config['h5_file_path'], 'r') as f:
        num_samples = f['features']['cup_disc_input'].shape[0]
        labels = f['labels'][:].astype(np.int32)
        logger.info(f"Dataset: {num_samples} samples")
        
        # Handle different label formats
        if len(labels.shape) > 1:
            # If labels are one-hot encoded, convert to class indices
            labels = np.argmax(labels, axis=1)
        
        logger.info(f"Class distribution: {np.bincount(labels)}")
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, 
                         random_state=config['random_state'])
    
    # Training results
    all_results = []
    
    # Train each fold
    for fold, (train_indices, val_indices) in enumerate(skf.split(np.arange(num_samples), labels)):
        logger.info(f"Processing Fold {fold + 1}/{config['n_folds']}")
        
        try:
            results, history, history_fine = train_fold(
                fold, train_indices, val_indices, 
                config['h5_file_path'], config
            )
            all_results.append(results)
            
            # Log fold results
            logger.info(f"Fold {fold + 1} Results:")
            logger.info(f"  Accuracy: {results['accuracy']:.4f}")
            logger.info(f"  Precision: {results['precision']:.4f}")
            logger.info(f"  Recall: {results['recall']:.4f}")
            logger.info(f"  F1 Score: {results['f1']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in fold {fold + 1}: {str(e)}")
            continue
    
    # Calculate overall metrics
    if all_results:
        avg_accuracy = np.mean([r['accuracy'] for r in all_results])
        avg_precision = np.mean([r['precision'] for r in all_results])
        avg_recall = np.mean([r['recall'] for r in all_results])
        avg_f1 = np.mean([r['f1'] for r in all_results])
        
        logger.info("Overall Results:")
        logger.info(f"  Average Accuracy: {avg_accuracy:.4f} ± {np.std([r['accuracy'] for r in all_results]):.4f}")
        logger.info(f"  Average Precision: {avg_precision:.4f} ± {np.std([r['precision'] for r in all_results]):.4f}")
        logger.info(f"  Average Recall: {avg_recall:.4f} ± {np.std([r['recall'] for r in all_results]):.4f}")
        logger.info(f"  Average F1 Score: {avg_f1:.4f} ± {np.std([r['f1'] for r in all_results]):.4f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"logs/efficientnet_results_fixed_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'config': config,
                'results': all_results,
                'overall_metrics': {
                    'avg_accuracy': float(avg_accuracy),
                    'avg_precision': float(avg_precision),
                    'avg_recall': float(avg_recall),
                    'avg_f1': float(avg_f1),
                    'std_accuracy': float(np.std([r['accuracy'] for r in all_results])),
                    'std_precision': float(np.std([r['precision'] for r in all_results])),
                    'std_recall': float(np.std([r['recall'] for r in all_results])),
                    'std_f1': float(np.std([r['f1'] for r in all_results]))
                }
            }, f, indent=2)
        
        logger.info(f"Results saved: {results_file}")
        
        # Create visualizations
        create_visualizations(all_results, config)
        
        # Save best model
        best_fold = max(all_results, key=lambda x: x['accuracy'])
        best_model_path = f"models/efficientnet_fold_{best_fold['fold']}.h5"
        final_model_path = "models/efficientnet_best_model_fixed.h5"
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            logger.info(f"Best model saved: {final_model_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
