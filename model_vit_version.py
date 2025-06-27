import os
import json
import numpy as np
import tensorflow as tf
import h5py
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, cohen_kappa_score,
    matthews_corrcoef
)
import psutil
import gc
from pathlib import Path
from data_generator import MemoryEfficientDataGenerator

# Enable mixed precision training
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
except Exception:
    pass

# Constants
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Model Architecture Constants
WINDOW_SIZE = 7
GRID_SIZE = 7
MAXVIT_DEPTHS = [2, 2, 5, 2]
MAXVIT_CHANNELS = [64, 128, 256, 512]
CONVNEXT_DEPTHS = [3, 3, 9, 3]
CONVNEXT_DIMS = [96, 192, 384, 768]
EXPANSION_RATIO = 4
NUM_HEADS = 8
DROPOUT_RATE = 0.1
STOCHASTIC_DEPTH_RATE = 0.1

# Training Constants
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
INITIAL_LR = 1e-4
MAX_LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5
TOTAL_EPOCHS = 100
RESTART_EPOCHS = 30

# Directory setup
for d in ['models', 'visualizations', 'logs']:
    os.makedirs(d, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/vit_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data keys and mapping
FEATURE_KEYS = [
    'rnfl_input', 'cup_disc_input', 'rim_input',
    'juxta_input', 'sheath_input', 'macular_input'
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

@dataclass
class ModelConfig:
    """Configuration for the hybrid model."""
    def __init__(
        self,
        input_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        window_size: int = 7,
        stochastic_depth_rate: float = 0.1
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.window_size = window_size
        self.stochastic_depth_rate = stochastic_depth_rate
        
        # Define transformer blocks configuration with compatible dimensions
        self.transformer_blocks = {
            'block1': {
                'dim': 64,
                'window_size': window_size,
                'num_heads': num_heads,
                'mlp_ratio': mlp_ratio,
                'drop_rate': drop_rate,
                'attn_drop_rate': attn_drop_rate,
                'drop_path_rate': drop_path_rate,
                'depth': 2
            },
            'block2': {
                'dim': 128,
                'window_size': window_size,
                'num_heads': num_heads,
                'mlp_ratio': mlp_ratio,
                'drop_rate': drop_rate,
                'attn_drop_rate': attn_drop_rate,
                'drop_path_rate': drop_path_rate,
                'depth': 2
            },
            'block3': {
                'dim': 256,
                'window_size': window_size,
                'num_heads': num_heads,
                'mlp_ratio': mlp_ratio,
                'drop_rate': drop_rate,
                'attn_drop_rate': attn_drop_rate,
                'drop_path_rate': drop_path_rate,
                'depth': 2
            }
        }

class MLP(tf.keras.layers.Layer):
    """MLP block for transformer blocks using Conv2D for 4D tensors."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
        act_layer: str = "gelu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fc1 = tf.keras.layers.Conv2D(hidden_features, 1, use_bias=True)
        self.act = tf.keras.layers.Activation(act_layer)
        self.fc2 = tf.keras.layers.Conv2D(out_features, 1, use_bias=True)
        self.drop = tf.keras.layers.Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RelativePositionBias(tf.keras.layers.Layer):
    """Relative position bias for attention."""
    def __init__(self, window_size: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.num_heads = num_heads
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * window_size - 1) * (2 * window_size - 1), num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='relative_position_bias_table'
        )
        
        coords_h = tf.range(window_size)
        coords_w = tf.range(window_size)
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = tf.reshape(coords, [2, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, [1, 2, 0])
        relative_coords = relative_coords + [window_size - 1, window_size - 1]
        relative_coords = relative_coords * [2 * window_size - 1, 1]
        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
            name='relative_position_index'
        )

    def call(self, x):
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, [-1])
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            [self.window_size * self.window_size, self.window_size * self.window_size, -1]
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        return relative_position_bias

class WindowAttention(tf.keras.layers.Layer):
    """Window attention layer."""
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        
        # Ensure dim is divisible by num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        self.head_dim = dim // num_heads
        
        # QKV projection
        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_dropout = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_dropout = tf.keras.layers.Dropout(proj_drop)
        
        # Relative position bias
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * window_size - 1) * (2 * window_size - 1), num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name='relative_position_bias_table'
        )
        
        # Get pair-wise relative position index
        coords_h = tf.range(window_size)
        coords_w = tf.range(window_size)
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = tf.reshape(coords, [2, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
        relative_coords = relative_coords + [window_size - 1, window_size - 1]
        relative_coords = relative_coords * [2 * window_size - 1, 1]
        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
            name='relative_position_index'
        )

    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, x, mask=None):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Pad input if necessary (use tf.cond for symbolic tensors)
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        should_pad = tf.logical_or(tf.not_equal(pad_h, 0), tf.not_equal(pad_w, 0))
        def pad_fn():
            return tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        x = tf.cond(should_pad, pad_fn, lambda: x)
        H = H + pad_h
        W = W + pad_w
        
        # Reshape to windows
        x = tf.reshape(x, [-1, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, self.window_size * self.window_size, C])
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [-1, self.window_size * self.window_size, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = tf.matmul(q, k, transpose_b=True) * (self.head_dim ** -0.5)
        
        # Add relative position bias
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, [-1])
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            [self.window_size * self.window_size, self.window_size * self.window_size, -1]
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)
        
        if mask is not None:
            attn = attn + mask
        
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn)
        
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, self.window_size * self.window_size, C])
        
        # Project back
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        # Reshape back
        x = tf.reshape(x, [-1, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, H, W, C])
        
        # Remove padding if it was added (use tf.cond)
        def unpad_fn():
            return x[:, :H-pad_h, :W-pad_w, :]
        x = tf.cond(should_pad, unpad_fn, lambda: x)
        
        return x

class MaxViTBlock(tf.keras.layers.Layer):
    """MaxViT block combining MBConv and window attention."""
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "gelu",
        norm_layer: str = "layer_norm",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        # MBConv block
        self.mbconv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim * 4, 1, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(act_layer),
            tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(act_layer),
            tf.keras.layers.Conv2D(dim, 1, use_bias=False),
            tf.keras.layers.BatchNormalization()
        ])

        # Projection shortcut for residual if needed
        self.proj_shortcut = None

        # Window attention
        self.window_attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        # MLP
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop,
            act_layer=act_layer
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.keras.layers.Lambda(lambda x: x)

    def build(self, input_shape):
        # If input channels != dim, add a projection shortcut
        in_channels = input_shape[-1]
        if in_channels != self.dim:
            self.proj_shortcut = tf.keras.layers.Conv2D(self.dim, 1, padding='same', use_bias=False)
        super().build(input_shape)

    def call(self, x):
        # MBConv block with projection shortcut if needed
        shortcut = x
        if self.proj_shortcut is not None:
            shortcut = self.proj_shortcut(shortcut)
        x = self.mbconv(x)
        x = shortcut + self.drop_path(x)

        # Window attention
        shortcut = x
        x = self.window_attn(x)
        x = shortcut + self.drop_path(x)

        # MLP
        shortcut = x
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)

        return x

class ConvNeXtBlock(tf.keras.layers.Layer):
    """ConvNeXt block with depth-wise convolution."""
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dwconv = tf.keras.layers.DepthwiseConv2D(
            7, padding='same', use_bias=False
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pwconv1 = tf.keras.layers.Dense(4 * dim)
        self.act = tf.keras.layers.Activation('gelu')
        self.pwconv2 = tf.keras.layers.Dense(dim)
        self.gamma = tf.Variable(
            initial_value=layer_scale_init_value * tf.ones((dim,)),
            trainable=True,
            name='gamma'
        )
        self.drop_path = tf.keras.layers.Dropout(drop_path)

    def call(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = shortcut + self.drop_path(x)
        return x

class HybridEncoder(tf.keras.layers.Layer):
    """Hybrid encoder combining MaxViT and ConvNeXt."""
    def __init__(
        self,
        dim: int,
        depth: int,
        window_size: int,
        num_heads: int,
        drop_path: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                MaxViTBlock(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    drop_rate=drop_path * i / depth
                )
            )
            self.blocks.append(
                ConvNeXtBlock(
                    dim=dim,
                    drop_path=drop_path * i / depth
                )
            )

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class CrossModalAttention(tf.keras.layers.Layer):
    """Cross-modal attention between different modalities."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

    def call(self, x, context):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = tf.transpose(
            tf.reshape(
                self.qkv(x),
                [-1, N, 3, self.num_heads, C // self.num_heads]
            ),
            [2, 0, 3, 1, 4]
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3])
        x = tf.reshape(x, [-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class HierarchicalFusion(tf.keras.layers.Layer):
    """Hierarchical feature fusion with adaptive weighting."""
    def __init__(
        self,
        dim: int,
        num_levels: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_levels = num_levels
        self.attention = CrossModalAttention(dim, num_heads=8)
        self.weight_layer = tf.keras.layers.Dense(num_levels, activation='softmax')
        self.fusion = tf.keras.layers.Dense(dim)

    def call(self, features):
        # Hierarchical attention
        fused = []
        for i in range(len(features)):
            attended = self.attention(features[i], features)
            fused.append(attended)
        
        # Adaptive weighting
        weights = self.weight_layer(tf.concat(fused, axis=-1))
        weighted = tf.reduce_sum(
            tf.stack(fused, axis=-1) * weights[..., None],
            axis=-1
        )
        
        return self.fusion(weighted)

class DropPath(tf.keras.layers.Layer):
    """Stochastic Depth regularization layer."""
    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if (not training) or (self.drop_prob == 0.0):
            return x
        keep_prob = 1.0 - self.drop_prob
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = len(x.shape)
        shape = [batch_size] + [1] * (rank - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = x / keep_prob * binary_tensor
        return output

class HybridGlaucomaClassifier(tf.keras.Model):
    """Hybrid model for multi-modal glaucoma classification."""
    def __init__(self, configur: ModelConfig):
        super().__init__()
        self.modalities = list(FEATURE_MAP.values())
        
        # Create a branch for each modality
        self.branches = {}
        for i, modality in enumerate(self.modalities):
            self.branches[modality] = tf.keras.Sequential([
                # Input preprocessing
                tf.keras.layers.Rescaling(1./255),
                
                # Initial convolution layers with dynamic padding
                tf.keras.layers.Conv2D(
                    32, 3, 
                    padding='same',
                    activation='relu',
                    kernel_initializer='he_normal'
                ),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.Conv2D(
                    64, 3,
                    padding='same',
                    activation='relu',
                    kernel_initializer='he_normal'
                ),
                tf.keras.layers.BatchNormalization(),
                
                # MaxViT block with dynamic window size
                MaxViTBlock(
                    dim=64,
                    window_size=configur.window_size,
                    num_heads=configur.num_heads,
                    mlp_ratio=configur.mlp_ratio,
                    drop=configur.drop_rate,
                    attn_drop=configur.attn_drop_rate,
                    drop_path=configur.drop_path_rate
                ),
                
                # Global pooling that works with any input size
                tf.keras.layers.GlobalAveragePooling2D()
            ])
            
        # Fusion layer
        self.fusion = tf.keras.layers.Concatenate()
        
        # Classifier with dropout for regularization
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(configur.num_classes, activation='softmax')
        ])
        
        # Input validation layer
        self.input_validation = tf.keras.layers.Lambda(
            lambda x: tf.debugging.assert_rank(x, 4, message="Input must be 4D tensor [batch, height, width, channels]")
        )

    def call(self, inputs, training=False):
        # Validate inputs
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary mapping modality names to tensors")
            
        features = []
        for modality in self.modalities:
            if modality not in inputs:
                raise ValueError(f"Missing input for modality: {modality}")
                
            x = inputs[modality]
            
            # Validate input shape
            self.input_validation(x)
            
            # Process through branch
            x = self.branches[modality](x, training=training)
            features.append(x)
            
        # Fuse features
        fused = self.fusion(features)
        
        # Final classification
        out = self.classifier(fused)
        return out
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "modalities": self.modalities
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(config=config["config"])

def create_hybrid_model(
    input_shape: Tuple[int, int, int] = IMG_SHAPE,
    num_classes: int = 5
) -> HybridGlaucomaClassifier:
    """Create and compile the hybrid model."""
    config = ModelConfig(
        input_size=input_shape[0],
        patch_size=input_shape[0] // 7,
        in_channels=input_shape[2],
        num_classes=num_classes,
        num_heads=8,
        embed_dim=768,
        depth=12
    )
    
    model = HybridGlaucomaClassifier(config)
    
    # Use AdamW optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=INITIAL_LR,
        weight_decay=WEIGHT_DECAY
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(
    model: HybridGlaucomaClassifier,
    train_generator: tf.keras.utils.Sequence,
    val_generator: tf.keras.utils.Sequence,
    epochs: int = TOTAL_EPOCHS
) -> tf.keras.callbacks.History:
    """Train the model with all optimizations."""
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=INITIAL_LR,
        first_decay_steps=RESTART_EPOCHS * len(train_generator),
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )
    model.optimizer.learning_rate = lr_schedule
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.CSVLogger('logs/training_history.csv')
    ]
    # Training
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    return history

def evaluate_model(
    model: HybridGlaucomaClassifier,
    test_generator: tf.keras.utils.Sequence
) -> Dict:
    """Evaluate the model with comprehensive metrics."""
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions[0], axis=1)
    y_true = test_generator.labels
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    # Calculate per-class metrics
    per_class_metrics = {
        'precision': precision_score(y_true, y_pred, average=None),
        'recall': recall_score(y_true, y_pred, average=None),
        'f1': f1_score(y_true, y_pred, average=None)
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save results
    results = {
        'metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist()
    }
    
    with open('logs/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def visualize_results(
    model: HybridGlaucomaClassifier,
    history: tf.keras.callbacks.History,
    test_generator: tf.keras.utils.Sequence
):
    """Create comprehensive visualizations."""
    # Training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['classifier_accuracy'], label='train_accuracy')
    plt.plot(history.history['val_classifier_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/training_history.png')
    plt.close()
    
    # Confusion matrix
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions[0], axis=1)
    y_true = test_generator.labels
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()

def save_model_package(
    model: HybridGlaucomaClassifier,
    history: tf.keras.callbacks.History,
    metrics: Dict,
    save_dir: str = 'models'
):
    """Save model in all required formats."""
    # Save full model
    model.save(f'{save_dir}/full_model.h5')
    
    # Save architecture
    model_json = model.to_json()
    with open(f'{save_dir}/model_architecture.json', 'w') as f:
        json.dump(model_json, f)
    
    # Save weights
    model.save_weights(f'{save_dir}/model_weights.h5')
    
    # Save training history
    with open(f'{save_dir}/training_history.json', 'w') as f:
        json.dump(history.history, f)
    
    # Save metrics
    with open(f'{save_dir}/model_metrics.json', 'w') as f:
        json.dump(metrics, f)

def main():
    """Main execution function."""
    try:
        # Load data
        h5_path = 'synthetic_glaucoma_data.h5'
        with h5py.File(h5_path, 'r') as f:
            features = {}
            for k in FEATURE_KEYS:
                arr = f['features'][k][:]
                arr = arr.astype(np.float32)
                arr = np.expand_dims(arr, axis=-1)
                features[FEATURE_MAP[k]] = arr
            labels = f['labels'][:]
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                class_indices = np.argmax(labels, axis=1)
            else:
                class_indices = labels.astype(int)
        
        logger.info(f"Loaded {len(class_indices)} samples")
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            np.arange(len(class_indices)),
            test_size=0.2,
            stratify=class_indices,
            random_state=SEED
        )
        
        # Create data generators using MemoryEfficientDataGenerator
        train_gen = MemoryEfficientDataGenerator(
            h5_file_path=h5_path,
            batch_size=BATCH_SIZE,
            feature_types=FEATURE_KEYS,
            shuffle=True,
            augment=True
        )
        test_gen = MemoryEfficientDataGenerator(
            h5_file_path=h5_path,
            batch_size=BATCH_SIZE,
            feature_types=FEATURE_KEYS,
            shuffle=False,
            augment=False
        )
        
        # Create and train model
        model = create_hybrid_model()
        
        # Train model
        history = train_model(model, train_gen, test_gen)
        
        # Evaluate model
        metrics = evaluate_model(model, test_gen)
        
        # Visualize results
        visualize_results(model, history, test_gen)
        
        # Save model package
        save_model_package(model, history, metrics)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main() 