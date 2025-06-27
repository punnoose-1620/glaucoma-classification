import tensorflow as tf
import logging
import os
from datetime import datetime
import json
import argparse

from data_generator import MemoryEfficientDataGenerator
from model_architectures import build_efficient_model, estimate_memory_usage
from memory_utils import MemoryMonitor, BatchSizeOptimizer, setup_memory_efficient_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientTrainer:
    def __init__(self, config_path=None):
        """
        Initialize memory-efficient trainer.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup memory-efficient training
        setup_memory_efficient_training()
        
        # Initialize memory monitoring
        self.memory_monitor = MemoryMonitor(log_dir=self.config['log_dir'])
        self.batch_optimizer = BatchSizeOptimizer(
            initial_batch_size=self.config['batch_size'],
            min_batch_size=self.config['min_batch_size'],
            max_batch_size=self.config['max_batch_size']
        )
        
        # Create model
        self.model = build_efficient_model(
            input_shape=self.config['input_shape'],
            num_classes=self.config['num_classes'],
            model_size=self.config['model_size'],
            use_mixed_precision=self.config['use_mixed_precision']
        )
        
        # Estimate memory usage
        memory_estimate = estimate_memory_usage(
            self.model,
            batch_size=self.config['batch_size']
        )
        # Convert all numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(v) for v in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return obj
        memory_estimate_py = convert_types(memory_estimate)
        logger.info(f"Estimated memory usage: {json.dumps(memory_estimate_py, indent=2)}")
    
    def _load_config(self, config_path):
        """Load configuration from file or use defaults"""
        default_config = {
            'data_path': 'synthetic_glaucoma_data.h5',
            'log_dir': 'logs',
            'checkpoint_dir': 'checkpoints',
            'batch_size': 32,
            'min_batch_size': 1,
            'max_batch_size': 64,
            'epochs': 100,
            'input_shape': (224, 224, 1),
            'num_classes': 5,
            'model_size': 'medium',
            'use_mixed_precision': True,
            'accumulation_steps': 8,
            'learning_rate': 1e-4,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'feature_types': ['RNFL', 'Cup-to-Disc', 'Rim', 
                            'Juxtapapillary', 'Sheath', 'Macular']
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def train(self):
        """Train the model with memory-efficient strategies"""
        # Create data generators
        train_generator = MemoryEfficientDataGenerator(
            h5_file_path=self.config['data_path'],
            batch_size=self.config['batch_size'],
            feature_types=self.config['feature_types'],
            shuffle=True,
            augment=True
        )
        
        val_generator = MemoryEfficientDataGenerator(
            h5_file_path=self.config['data_path'],
            batch_size=self.config['batch_size'],
            feature_types=self.config['feature_types'],
            shuffle=False,
            augment=False
        )
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config['checkpoint_dir'], 'model_{epoch:02d}.h5'),
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.config['log_dir'], 'tensorboard')
            )
        ]
        
        # Training loop with memory monitoring
        try:
            for epoch in range(self.config['epochs']):
                logger.info(f"Starting epoch {epoch + 1}/{self.config['epochs']}")
                
                # Monitor memory before epoch
                self.memory_monitor.log_memory_usage(phase='epoch_start')
                
                # Train for one epoch
                history = self.model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=1,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Monitor memory after epoch
                self.memory_monitor.log_memory_usage(phase='epoch_end')
                
                # Adjust batch size if needed
                new_batch_size = self.batch_optimizer.adjust_batch_size()
                if new_batch_size != self.config['batch_size']:
                    self.config['batch_size'] = new_batch_size
                    train_generator.batch_size = new_batch_size
                    val_generator.batch_size = new_batch_size
                    logger.info(f"Batch size adjusted to {new_batch_size}")
                
                # Save memory log periodically
                if (epoch + 1) % 5 == 0:
                    self.memory_monitor.save_memory_log()
        
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"Out of memory error: {e}")
            self._handle_oom_error()
        
        finally:
            # Save final memory log
            self.memory_monitor.save_memory_log()
            
            # Save model
            self.model.save(os.path.join(self.config['checkpoint_dir'], 'final_model.h5'))
            logger.info("Training completed and model saved")
    
    def _handle_oom_error(self):
        """Handle out-of-memory error"""
        logger.info("Attempting to recover from OOM error...")
        
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Reduce batch size
        self.config['batch_size'] = max(1, self.config['batch_size'] // 2)
        logger.info(f"Reduced batch size to {self.config['batch_size']}")
        
        # Save emergency checkpoint
        self.model.save(os.path.join(self.config['checkpoint_dir'], 'emergency_checkpoint.h5'))
        logger.info("Emergency checkpoint saved")

def main():
    parser = argparse.ArgumentParser(description='Memory-efficient glaucoma classification training')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    
    trainer = MemoryEfficientTrainer(config_path=args.config)
    trainer.train()

if __name__ == '__main__':
    main() 