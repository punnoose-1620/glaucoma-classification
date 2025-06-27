import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import h5py
import kaggle
from datetime import datetime
import logging
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.utils.class_weight import compute_class_weight
# from grad_cam import GradCAM

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_file, indices, feature_types, batch_size, img_size, num_classes, shuffle=True):
        self.h5_file = h5_file
        self.indices = indices
        self.feature_types = feature_types
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.indices) // self.batch_size
        
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Initialize batch arrays
        X_batch = [[] for _ in self.feature_types]
        y_batch = []
        
        with h5py.File(self.h5_file, 'r') as f:
            # Load batch data
            for i in batch_indices:
                for j, feature_type in enumerate(self.feature_types):
                    img = f['features'][feature_type][i]
                    img = img.astype('float32') / 255.0
                    img = img[..., np.newaxis]
                    X_batch[j].append(img)
                y_batch.append(f['labels'][i])
        
        # Convert to numpy arrays
        X_batch = [np.array(x) for x in X_batch]
        y_batch = np.array(y_batch)
        
        # Convert to TensorFlow tensors
        X_batch = [tf.convert_to_tensor(x, dtype=tf.float32) for x in X_batch]
        y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
        
        return tuple(X_batch), y_batch
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def get_config(self):
        return {
            'h5_file': self.h5_file,
            'indices': self.indices,
            'feature_types': self.feature_types,
            'batch_size': self.batch_size,
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'shuffle': self.shuffle
        }
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class GlaucomaClassifier:
    def __init__(self):
        self.feature_types = [
            'rnfl_input', 'cup_disc_input', 'rim_input',
            'juxta_input', 'sheath_input', 'macular_input'
        ]
        self.classes = [
            'Normal', 'Open-angle', 'Angle-closure',
            'Normal-tension', 'Secondary'
        ]
        self.img_size = (224, 224)
        self.batch_size = 16  # Reduced batch size
        self.epochs = 100
        self.initial_learning_rate = 0.0001
        
        # Create necessary directories
        for dir_name in ['visualizations', 'logs', 'models']:
            os.makedirs(dir_name, exist_ok=True)
            
        # Initialize model
        self.model = None
        
    def load_dataset(self):
        """Load dataset from Kaggle using the API"""
        try:
            logging.info("Downloading dataset from Kaggle...")
            kaggle.api.dataset_download_files(
                'aleksanderprudnik/synthetic-oct-glaucoma-dataset',
                path='.',
                quiet=False,
                unzip=True
            )
            logging.info("Dataset downloaded successfully")
        except Exception as e:
            logging.error(f"Error downloading dataset: {str(e)}")
            raise
            
    def explore_data(self, h5_file_path):
        """Explore the structure of the HDF5 file"""
        with h5py.File(h5_file_path, 'r') as f:
            logging.info("Dataset structure:")
            for key in f.keys():
                logging.info(f"Group: {key}")
                if isinstance(f[key], h5py.Group):
                    for subkey in f[key].keys():
                        logging.info(f"  - {subkey}: {f[key][subkey].shape}")
                        
    def visualize_samples(self, h5_file_path):
        """Create visualizations of sample images from each feature type"""
        with h5py.File(h5_file_path, 'r') as f:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for idx, feature_type in enumerate(self.feature_types):
                # Get a random sample
                sample_idx = np.random.randint(0, 500)
                img = f['features'][feature_type][sample_idx]
                
                axes[idx].imshow(img, cmap='gray')
                axes[idx].set_title(feature_type)
                axes[idx].axis('off')
                
            plt.tight_layout()
            plt.savefig('visualizations/sample_images.png')
            plt.close()
            
    def prepare_data_generators(self, h5_file_path):
        """Prepare data generators for training, validation, and testing"""
        with h5py.File(h5_file_path, 'r') as f:
            # Get total number of samples
            num_samples = f['labels'].shape[0]
            indices = np.arange(num_samples)
            
            # Calculate class weights
            y = f['labels'][:]
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(np.argmax(y, axis=1)),
                y=np.argmax(y, axis=1)
            )
            self.class_weights = dict(enumerate(class_weights))
            
            # Split indices
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.3, random_state=42,
                stratify=np.argmax(y, axis=1)
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=42,
                stratify=np.argmax(y[temp_idx], axis=1)
            )
            
            # Create data generators
            train_generator = DataGenerator(
                h5_file_path, train_idx, self.feature_types,
                self.batch_size, self.img_size, len(self.classes), shuffle=True
            )
            
            val_generator = DataGenerator(
                h5_file_path, val_idx, self.feature_types,
                self.batch_size, self.img_size, len(self.classes), shuffle=False
            )
            
            test_generator = DataGenerator(
                h5_file_path, test_idx, self.feature_types,
                self.batch_size, self.img_size, len(self.classes), shuffle=False
            )
            
            return train_generator, val_generator, test_generator
            
    def build_model(self):
        """Build the improved multi-input CNN model"""
        # Input layers for each feature type
        inputs = []
        for _ in self.feature_types:
            inputs.append(layers.Input(shape=(*self.img_size, 1)))
            
        # Process each input through a CNN branch
        branches = []
        for input_tensor in inputs:
            # First block
            x = layers.Conv2D(64, (3, 3), padding='same')(input_tensor)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(64, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Second block
            x = layers.Conv2D(128, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(128, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Third block
            x = layers.Conv2D(256, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(256, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Flatten()(x)
            branches.append(x)
            
        # Concatenate all branches
        merged = layers.Concatenate()(branches)
        
        # Dense layers with residual connections
        x = layers.Dense(512)(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        residual = x
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Add()([x, residual])
        
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(len(self.classes), activation='softmax')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model with AdamW optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.initial_learning_rate,
            weight_decay=0.001
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logging.info("Model built successfully")
        
    def train_model(self, train_generator, val_generator):
        """Train the model with improved callbacks"""
        # Learning rate schedule
        initial_learning_rate = self.initial_learning_rate
        total_steps = len(train_generator) * self.epochs
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=total_steps
        )
        
        # Update optimizer with schedule
        self.model.optimizer.learning_rate = lr_schedule
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs/tensorboard',
                histogram_freq=1
            ),
            # Add logging callback
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: logging.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"loss: {logs['loss']:.4f} - "
                    f"accuracy: {logs['accuracy']:.4f} - "
                    f"val_loss: {logs['val_loss']:.4f} - "
                    f"val_accuracy: {logs['val_accuracy']:.4f}"
                )
            )
        ]
        
        # Train model with class weights
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=self.class_weights
        )
        
        # Save training history
        with open('logs/training_history.json', 'w') as f:
            json.dump(history.history, f)
            
        return history
        
    def evaluate_model(self, test_generator):
        """Evaluate model performance on test data"""
        # Get predictions
        y_pred = []
        y_true = []
        
        for i in range(len(test_generator)):
            X_batch, y_batch = test_generator[i]
            y_pred_batch = self.model.predict(X_batch)
            y_pred.extend(y_pred_batch)
            y_true.extend(y_batch)
            
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('visualizations/confusion_matrix.png')
        plt.close()
        
        # Generate classification report
        report = classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=self.classes,
            output_dict=True
        )
        
        # Save report
        with open('logs/classification_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        return report
        
    def save_model(self):
        """Save model in multiple formats"""
        # Save TensorFlow model
        self.model.save('models/glaucoma_classifier.h5')
        
        # Save model architecture
        model_json = self.model.to_json()
        with open('models/model_architecture.json', 'w') as f:
            json.dump(model_json, f)
            
        logging.info("Model saved successfully")
        
def main():
    # Initialize classifier
    classifier = GlaucomaClassifier()
    
    # Load dataset
    classifier.load_dataset()
    
    # Explore data
    classifier.explore_data('synthetic_glaucoma_data.h5')
    
    # Create visualizations
    classifier.visualize_samples('synthetic_glaucoma_data.h5')
    
    # Prepare data generators
    train_generator, val_generator, test_generator = classifier.prepare_data_generators('synthetic_glaucoma_data.h5')
    
    # Build and train model
    classifier.build_model()
    history = classifier.train_model(train_generator, val_generator)
    
    # Evaluate model
    report = classifier.evaluate_model(test_generator)
    
    # Save model
    classifier.save_model()
    
if __name__ == "__main__":
    main() 