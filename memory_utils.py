import psutil
import gc
import tensorflow as tf
import logging
import json
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    def __init__(self, log_dir='logs'):
        """
        Initialize memory monitor.
        
        Args:
            log_dir (str): Directory to save memory logs
        """
        self.log_dir = log_dir
        self.memory_log = []
        self.start_time = datetime.now()
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set memory thresholds (in MB)
        self.warning_threshold = 0.8  # 80% of available memory
        self.critical_threshold = 0.9  # 90% of available memory
        
        # Get system memory info
        self.total_memory = psutil.virtual_memory().total / (1024 * 1024)  # Convert to MB
        logger.info(f"Total system memory: {self.total_memory:.2f} MB")
    
    def get_memory_usage(self):
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_rss': memory_info.rss / (1024 * 1024),  # MB
            'process_vms': memory_info.vms / (1024 * 1024),  # MB
            'system_used': system_memory.used / (1024 * 1024),  # MB
            'system_percent': system_memory.percent,
            'timestamp': datetime.now().isoformat()
        }
    
    def log_memory_usage(self, phase='training'):
        """Log current memory usage"""
        memory_stats = self.get_memory_usage()
        memory_stats['phase'] = phase
        self.memory_log.append(memory_stats)
        
        # Check thresholds
        if memory_stats['system_percent'] > self.critical_threshold * 100:
            logger.warning(f"CRITICAL: Memory usage at {memory_stats['system_percent']}%")
            self._emergency_cleanup()
        elif memory_stats['system_percent'] > self.warning_threshold * 100:
            logger.warning(f"WARNING: Memory usage at {memory_stats['system_percent']}%")
    
    def save_memory_log(self):
        """Save memory usage log to file"""
        log_file = os.path.join(
            self.log_dir,
            f'memory_usage_{self.start_time.strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(log_file, 'w') as f:
            json.dump(self.memory_log, f, indent=2)
        
        logger.info(f"Memory log saved to {log_file}")
    
    def _emergency_cleanup(self):
        """Perform emergency memory cleanup"""
        logger.info("Performing emergency memory cleanup...")
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()
        
        # Log memory after cleanup
        self.log_memory_usage(phase='emergency_cleanup')

class BatchSizeOptimizer:
    def __init__(self, initial_batch_size=32, min_batch_size=1, max_batch_size=64):
        """
        Initialize batch size optimizer.
        
        Args:
            initial_batch_size (int): Initial batch size
            min_batch_size (int): Minimum allowed batch size
            max_batch_size (int): Maximum allowed batch size
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_monitor = MemoryMonitor()
    
    def adjust_batch_size(self, memory_threshold=0.8):
        """
        Adjust batch size based on memory usage.
        
        Args:
            memory_threshold (float): Memory threshold (0-1) for batch size adjustment
        
        Returns:
            int: New batch size
        """
        memory_stats = self.memory_monitor.get_memory_usage()
        
        if memory_stats['system_percent'] > memory_threshold * 100:
            # Reduce batch size
            new_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.5)
            )
            logger.info(f"Reducing batch size from {self.current_batch_size} to {new_batch_size}")
        else:
            # Try to increase batch size
            new_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.5)
            )
            logger.info(f"Increasing batch size from {self.current_batch_size} to {new_batch_size}")
        
        self.current_batch_size = new_batch_size
        return new_batch_size

def setup_memory_efficient_training():
    """Configure TensorFlow for memory-efficient training"""
    # Limit GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled")
        except RuntimeError as e:
            logger.error(f"Error setting GPU memory growth: {e}")
    
    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    logger.info("Mixed precision training enabled")
    
    # Configure TensorFlow for memory efficiency
    tf.config.optimizer.set_jit(True)  # Enable XLA
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': True,
        'constant_folding': True,
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
        'dependency_optimization': True,
        'loop_optimization': True,
        'function_optimization': True,
        'debug_stripper': True,
        'disable_model_pruning': False,
        'scoped_allocator_optimization': True,
        'pin_to_host_optimization': True,
        'implementation_selector': True,
        'auto_mixed_precision': True
    })
    logger.info("TensorFlow memory optimizations enabled") 