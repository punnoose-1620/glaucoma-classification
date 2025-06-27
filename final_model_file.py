import os
import json
import numpy as np
import tensorflow as tf
import h5py
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import psutil
import gc
from pathlib import Path

# Enable mixed precision training
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
except Exception:
    pass

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Directory setup
for d in ['models', 'visualizations', 'logs']:
    os.makedirs(d, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/run.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Data keys and mapping (from data_generator.py)
FEATURE_KEYS = [
    'rnfl_input', 'cup_disc_input', 'rim_input', 'juxta_input', 'sheath_input', 'macular_input'
]
FEATURE_MAP = {
    'rnfl_input': 'RNFL',
    'cup_disc_input': 'Cup-to-Disc',
    'rim_input': 'Rim',
    'juxta_input': 'Juxtapapillary',
    'sheath_input': 'Sheath',
    'macular_input': 'Macular',
}
IMG_SHAPE = (224, 224, 1)

# Data loading and generator (reference: data_generator.py)
def load_h5_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        features = {}
        for k in FEATURE_KEYS:
            arr = f['features'][k][:]
            arr = arr.astype(np.float32)
            arr = np.expand_dims(arr, axis=-1)  # (N, 224, 224, 1)
            features[FEATURE_MAP[k]] = arr
        labels = f['labels'][:]
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            class_indices = np.argmax(labels, axis=1)
        else:
            class_indices = labels.astype(int)
        logger.info(f"Loaded {len(class_indices)} samples. Class distribution: {np.bincount(class_indices)}")
        return features, class_indices

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, features, labels, indices, batch_size=32, augment=False):
        self.features = features
        self.labels = labels
        self.indices = indices
        self.batch_size = batch_size
        self.augment = augment
        self.feature_names = list(features.keys())
        self.on_epoch_end()
        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = {k: self.features[k][batch_idx] for k in self.feature_names}
        if self.augment:
            for k in batch_x:
                batch_x[k] = self.augmentation(batch_x[k])
        batch_y = self.labels[batch_idx]
        return batch_x, batch_y
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# Advanced Model Architecture (multi-input, attention, fusion, etc.)
def squeeze_excite_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    return tf.keras.layers.Multiply()([input_tensor, se])

def inception_module(x, filters):
    path1 = tf.keras.layers.Conv2D(filters, 1, padding='same', activation='relu')(x)
    path2 = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    path3 = tf.keras.layers.Conv2D(filters, 5, padding='same', activation='relu')(x)
    path4 = tf.keras.layers.MaxPooling2D(3, strides=1, padding='same')(x)
    path4 = tf.keras.layers.Conv2D(filters, 1, padding='same', activation='relu')(path4)
    return tf.keras.layers.Concatenate()([path1, path2, path3, path4])

def build_branch(input_shape, backbone='resnet', name=None):
    inp = tf.keras.layers.Input(shape=input_shape, name=name)
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return inp, x

def multihead_attention_fusion(feature_list, d_model=120, num_heads=4):
    concat = tf.keras.layers.Concatenate()(feature_list)
    x = tf.keras.layers.Dense(d_model, activation='relu')(concat)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((len(feature_list), d_model // len(feature_list)))(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // len(feature_list))
    x = attn(x, x)
    x = tf.keras.layers.Flatten()(x)
    return x

def build_advanced_model(input_shapes, num_classes=2):
    inputs = []
    features = []
    for k in FEATURE_MAP.values():
        inp, feat = build_branch(input_shapes[k], name=k)
        inputs.append(inp)
        features.append(feat)
    
    # Attention fusion
    fusion = multihead_attention_fusion(features)
    x = tf.keras.layers.Dense(256, activation='relu')(fusion)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=out)
    
    # Use AdamW optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Cross-validation, training, evaluation, and logging
def run_cross_validation(features, labels, n_splits=5, batch_size=16, epochs=50):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    input_shapes = {k: IMG_SHAPE for k in FEATURE_MAP.values()}
    all_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"Fold {fold+1}/{n_splits}")
        train_gen = DataGenerator(features, labels, train_idx, batch_size=batch_size, augment=True)
        val_gen = DataGenerator(features, labels, val_idx, batch_size=batch_size, augment=False)
        
        model = build_advanced_model(input_shapes, num_classes=len(np.unique(labels)))
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'models/best_model_fold_{fold+1}.h5',
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.CSVLogger(f'logs/training_history_fold_{fold+1}.csv')
        ]
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )
        
        # Evaluation
        val_preds = model.predict(val_gen)
        val_pred_labels = np.argmax(val_preds, axis=1)
        
        acc = accuracy_score(labels[val_idx], val_pred_labels)
        prec = precision_score(labels[val_idx], val_pred_labels, average='weighted', zero_division=0)
        rec = recall_score(labels[val_idx], val_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(labels[val_idx], val_pred_labels, average='weighted', zero_division=0)
        auc = roc_auc_score(
            labels[val_idx],
            tf.keras.utils.to_categorical(labels[val_idx], num_classes=len(np.unique(labels))),
            multi_class='ovr'
        )
        cm = confusion_matrix(labels[val_idx], val_pred_labels)
        
        metrics = {
            'fold': fold+1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist()
        }
        all_metrics.append(metrics)
        
        # Save confusion matrix
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix Fold {fold+1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'visualizations/confusion_matrix_fold_{fold+1}.png')
        plt.close()
        
        # Save training history plot
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.title('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'visualizations/training_history_fold_{fold+1}.png')
        plt.close()
        
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()
    
    # Save metrics
    with open('logs/cross_validation_results.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"Cross-validation complete. Metrics: {all_metrics}")
    return all_metrics

def main():
    h5_path = 'synthetic_glaucoma_data.h5'
    features, labels = load_h5_data(h5_path)
    run_cross_validation(features, labels, n_splits=5, batch_size=16, epochs=50)

if __name__ == '__main__':
    main() 