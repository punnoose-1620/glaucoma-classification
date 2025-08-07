#!/usr/bin/env python3
"""
Force GPU Training Script for NVIDIA RTX 4060
This script forces TensorFlow to use the NVIDIA GPU even if not properly detected
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import logging
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import psutil
import gc

# Force GPU configuration
def force_gpu_config():
    """Force TensorFlow to use NVIDIA GPU"""
    print("="*60)
    print("FORCING GPU CONFIGURATION FOR NVIDIA RTX 4060")
    print("="*60)
    
    # Set environment variables to force GPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU (NVIDIA)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Configure TensorFlow to use GPU
    try:
        # List all physical devices
        physical_devices = tf.config.list_physical_devices()
        print(f"All physical devices: {physical_devices}")
        
        # Try to configure GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s): {gpus}")
            
            # Enable memory growth for all GPUs
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✓ Memory growth enabled for {gpu}")
                except RuntimeError as e:
                    print(f"⚠️  Warning: {e}")
            
            # Set mixed precision for better performance
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("✓ Mixed precision enabled")
            except Exception as e:
                print(f"⚠️  Warning: Could not enable mixed precision: {e}")
            
            # Enable XLA optimization
            tf.config.optimizer.set_jit(True)
            print("✓ XLA optimization enabled")
            
            # Set device placement strategy
            tf.config.set_soft_device_placement(True)
            print("✓ Soft device placement enabled")
            
            return True
        else:
            print("⚠️  No GPU detected by TensorFlow")
            print("   This may be due to missing CUDA support")
            return False
            
    except Exception as e:
        print(f"✗ Error configuring GPU: {e}")
        return False

def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"force_gpu_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate focal loss
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_loss = self.alpha * tf.pow(1 - p_t, self.gamma) * cross_entropy
        
        return tf.reduce_mean(focal_loss)

class ForceGPUDataGenerator(keras.utils.Sequence):
    """Data generator that forces GPU usage"""
    
    def __init__(self, h5_file, batch_size=32, shuffle=True, augment=True):
        self.h5_file = h5_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
        with h5py.File(h5_file, 'r') as f:
            self.num_samples = len(f['labels'])
            self.indices = np.arange(self.num_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_batch(self, batch_indices):
        """Load a batch of data"""
        with h5py.File(self.h5_file, 'r') as f:
            # Load labels
            labels = f['labels'][batch_indices]
            
            # Load all feature types and combine them
            feature_types = ['rnfl_input', 'cup_disc_input', 'rim_input', 
                           'juxta_input', 'sheath_input', 'macular_input']
            
            combined_features = []
            for feature_type in feature_types:
                if feature_type in f['features']:
                    feature_data = f['features'][feature_type][batch_indices]
                    combined_features.append(feature_data)
            
            # Combine all features along the channel dimension
            if combined_features:
                # Stack features along the last dimension (channels)
                combined_inputs = np.stack(combined_features, axis=-1)
            else:
                # Fallback: use first available feature
                first_feature = list(f['features'].keys())[0]
                combined_inputs = f['features'][first_feature][batch_indices]
                # Add channel dimension
                combined_inputs = np.expand_dims(combined_inputs, axis=-1)
        
        # Normalize to [0, 1] range
        combined_inputs = np.clip(combined_inputs, 0, 255) / 255.0
        
        # Convert to float32 for GPU efficiency
        combined_inputs = combined_inputs.astype(np.float32)
        
        # Labels are already one-hot encoded, just return them as is
        return combined_inputs, labels
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Sort indices to ensure they're in increasing order for h5py
        batch_indices = np.sort(batch_indices)
        
        return self._load_batch(batch_indices)

def create_force_gpu_model():
    """Create a model that forces GPU usage"""
    print("Creating model with forced GPU usage...")
    
    # Force GPU device placement
    with tf.device('/GPU:0'):
        model = keras.Sequential([
            # Input layer - 6 channels from combined features
            layers.Input(shape=(224, 224, 6)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(5, activation='softmax')
        ])
        
        # Compile with GPU-optimized settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=FocalLoss(),
            metrics=['accuracy']
        )
    
    print(f"Model created with {model.count_params():,} parameters")
    return model

def create_callbacks():
    """Create training callbacks"""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='models/force_gpu_model_fold_{epoch:02d}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks

def train_fold(fold, train_generator, val_generator, logger):
    """Train a single fold with forced GPU usage"""
    logger.info(f"Starting training for Fold {fold} with forced GPU")
    
    # Create model with forced GPU
    model = create_force_gpu_model()
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train the model with GPU device placement
    with tf.device('/GPU:0'):
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=50,
            callbacks=callbacks,
            verbose=1
        )
    
    # Save the final model
    model.save(f'models/force_gpu_model_fold_{fold}_final.h5')
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    logger.info(f"Fold {fold} - Validation Accuracy: {val_accuracy:.4f}")
    
    return model, history, val_accuracy

def create_visualizations(history, fold, logger):
    """Create training visualizations"""
    logger.info(f"Creating visualizations for Fold {fold}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training history
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'Model Accuracy - Fold {fold}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'Model Loss - Fold {fold}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/force_gpu_training_history_fold_{fold}.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main training function with forced GPU usage"""
    print("="*60)
    print("FORCE GPU GLAUCOMA CLASSIFICATION TRAINING")
    print("="*60)
    
    # Force GPU configuration
    gpu_available = force_gpu_config()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting force GPU training")
    
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
        
        # Create data generators
        train_generator = ForceGPUDataGenerator(
            "synthetic_glaucoma_data.h5",
            batch_size=32,
            shuffle=True,
            augment=True
        )
        
        val_generator = ForceGPUDataGenerator(
            "synthetic_glaucoma_data.h5",
            batch_size=32,
            shuffle=False,
            augment=False
        )
        
        # Train the fold
        model, history, val_accuracy = train_fold(fold, train_generator, val_generator, logger)
        
        # Create visualizations
        create_visualizations(history, fold, logger)
        
        # Store results
        fold_accuracies.append(val_accuracy)
        
        # Clear memory
        del model
        gc.collect()
        
        # Log memory usage
        memory_usage = psutil.virtual_memory().percent
        logger.info(f"Memory usage: {memory_usage:.1f}%")
    
    # Final results
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Average accuracy across {n_folds} folds: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    logger.info(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    
    print(f"\nTraining completed! Average accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Models saved in 'models/' directory")
    print(f"Visualizations saved in 'visualizations/' directory")

if __name__ == "__main__":
    main()
