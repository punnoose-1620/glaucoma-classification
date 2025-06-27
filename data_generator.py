import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import logging
import psutil
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryEfficientDataGenerator(Sequence):
    def __init__(self, h5_file_path, batch_size=32, feature_types=None, 
                 shuffle=True, augment=True, chunk_size=32):
        """
        Memory-efficient data generator for glaucoma classification.
        
        Args:
            h5_file_path (str): Path to HDF5 file
            batch_size (int): Batch size for training
            feature_types (list): List of feature types to use
            shuffle (bool): Whether to shuffle data
            augment (bool): Whether to apply data augmentation
            chunk_size (int): HDF5 chunk cache size in MB
        """
        self.h5_file_path = h5_file_path
        self.batch_size = batch_size
        self.feature_types = feature_types or [
            'rnfl_input', 'cup_disc_input', 'rim_input', 'juxta_input', 'sheath_input', 'macular_input'
        ]
        self.shuffle = shuffle
        self.augment = augment
        self.chunk_size = chunk_size * 1024 * 1024  # Convert MB to bytes
        
        # Initialize HDF5 file with chunked access
        self.h5_file = h5py.File(h5_file_path, 'r', rdcc_nbytes=self.chunk_size)
        
        # Get dataset dimensions
        self.num_samples = len(self.h5_file['features'][self.feature_types[0]])
        self.image_shape = self.h5_file['features'][self.feature_types[0]][0].shape
        
        # Create index mapping
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        # Initialize augmentation
        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])
        
        # Mapping from HDF5 keys to model input names
        self.feature_map = {
            'rnfl_input': 'RNFL',
            'cup_disc_input': 'Cup-to-Disc',
            'rim_input': 'Rim',
            'juxta_input': 'Juxtapapillary',
            'sheath_input': 'Sheath',
            'macular_input': 'Macular',
        }
        
        logger.info(f"Initialized data generator with {self.num_samples} samples")
        self._log_memory_usage()
    
    def __len__(self):
        """Returns number of batches per epoch"""
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, idx):
        """Generate one batch of data"""
        try:
            # Get batch indices
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            # Initialize batch arrays
            batch_x = {feature: [] for feature in self.feature_types}
            batch_y = []
            
            # Load data in chunks
            for i in batch_indices:
                # Load features
                for feature in self.feature_types:
                    img = self.h5_file['features'][feature][i]
                    img = img.astype(np.float32)
                    img = np.expand_dims(img, axis=-1)  # (224,224) -> (224,224,1)
                    if self.augment:
                        img = self.augmentation(img)
                    batch_x[feature].append(img)
                
                # Load labels
                label = self.h5_file['labels'][i]
                batch_y.append(label.astype(np.float32))
            
            # Convert to numpy arrays
            batch_x = {k: np.stack(v, axis=0) for k, v in batch_x.items()}
            batch_y = np.stack(batch_y, axis=0)
            # Map to model input names
            batch_x = {self.feature_map[k]: v for k, v in batch_x.items()}
            
            # Ensure labels are always integer class indices
            if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                batch_y = np.argmax(batch_y, axis=1)
            return batch_x, batch_y
            
        except Exception as e:
            logger.error(f"Error in batch generation: {str(e)}")
            self._emergency_memory_recovery()
            raise
    
    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        self._log_memory_usage()
    
    def _log_memory_usage(self):
        """Log current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    def _emergency_memory_recovery(self):
        """Emergency memory cleanup"""
        gc.collect()
        tf.keras.backend.clear_session()
        self._log_memory_usage()
    
    def __del__(self):
        """Cleanup when generator is destroyed"""
        if hasattr(self, 'h5_file'):
            self.h5_file.close() 