import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, use_bias=False):
        super(MemoryEfficientBlock, self).__init__()
        self.depthwise = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=use_bias
        )
        self.bn1 = layers.BatchNormalization()
        self.pointwise = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=use_bias
        )
        self.bn2 = layers.BatchNormalization()
        self.activation = layers.ReLU()
        
    def call(self, inputs, training=None):
        x = self.depthwise(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.pointwise(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        return x

def build_efficient_model(input_shape=(224, 224, 1), num_classes=5, 
                         model_size='medium', use_mixed_precision=True):
    """
    Build a memory-efficient model for glaucoma classification.
    
    Args:
        input_shape (tuple): Input image shape
        num_classes (int): Number of output classes
        model_size (str): Model size ('small', 'medium', 'large')
        use_mixed_precision (bool): Whether to use mixed precision training
    
    Returns:
        tf.keras.Model: Compiled model
    """
    # Configure mixed precision if requested
    if use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision training enabled")
    
    # Model size configurations
    size_configs = {
        'small': {'filters': [32, 64, 128], 'blocks': [2, 2, 2]},
        'medium': {'filters': [64, 128, 256], 'blocks': [3, 3, 3]},
        'large': {'filters': [128, 256, 512], 'blocks': [4, 4, 4]}
    }
    
    config = size_configs[model_size]
    
    # Input layers for each feature type
    inputs = {
        'RNFL': layers.Input(shape=input_shape, name='RNFL'),
        'Cup-to-Disc': layers.Input(shape=input_shape, name='Cup-to-Disc'),
        'Rim': layers.Input(shape=input_shape, name='Rim'),
        'Juxtapapillary': layers.Input(shape=input_shape, name='Juxtapapillary'),
        'Sheath': layers.Input(shape=input_shape, name='Sheath'),
        'Macular': layers.Input(shape=input_shape, name='Macular')
    }
    
    # Process each input separately
    feature_outputs = []
    for feature_name, feature_input in inputs.items():
        x = feature_input
        
        # Initial convolution
        x = layers.Conv2D(config['filters'][0], 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Memory-efficient blocks
        for i, (filters, num_blocks) in enumerate(zip(config['filters'], config['blocks'])):
            for j in range(num_blocks):
                strides = 2 if j == 0 and i > 0 else 1
                x = MemoryEfficientBlock(filters, strides=strides)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        feature_outputs.append(x)
    
    # Combine features
    x = layers.Concatenate()(feature_outputs)
    
    # Final classification layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with gradient accumulation support
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Log model summary
    model.summary(print_fn=logger.info)
    
    return model

def estimate_memory_usage(model, batch_size=32):
    """
    Estimate memory usage for the model.
    
    Args:
        model (tf.keras.Model): The model to analyze
        batch_size (int): Batch size for estimation
    
    Returns:
        dict: Memory usage estimates
    """
    # Get model parameters
    trainable_params = sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = sum([np.prod(v.shape) for v in model.non_trainable_weights])
    
    # Estimate memory usage
    param_memory = (trainable_params + non_trainable_params) * 4  # 4 bytes per float32
    activation_memory = sum([
        np.prod(layer.output_shape[1:])
        for layer in model.layers
        if hasattr(layer, 'output_shape') and isinstance(layer.output_shape, tuple)
    ]) * batch_size * 4
    
    return {
        'parameters': {
            'trainable': trainable_params,
            'non_trainable': non_trainable_params,
            'total': trainable_params + non_trainable_params
        },
        'memory_usage': {
            'parameters_mb': param_memory / (1024 * 1024),
            'activations_mb': activation_memory / (1024 * 1024),
            'total_mb': (param_memory + activation_memory) / (1024 * 1024)
        }
    } 