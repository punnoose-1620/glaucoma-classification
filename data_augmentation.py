import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import albumentations as A
from scipy import ndimage
import random
import math


class MedicalImageAugmentation:
    """
    Comprehensive data augmentation pipeline for medical images (OCT scans)
    """
    
    def __init__(self, image_size=(224, 224), p=0.5):
        """
        Initialize augmentation pipeline
        
        Args:
            image_size: Target image size (height, width)
            p: Probability of applying augmentations
        """
        self.image_size = image_size
        self.p = p
        
        # Initialize Albumentations pipeline
        self.albumentations_pipeline = A.Compose([
            # Geometric transformations
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15, p=0.3),
            
            # Color and contrast augmentations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.MedianBlur(blur_limit=5, p=0.2),
            
            # Medical-specific augmentations
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ], p=1.0)
    
    def augment_single_image(self, image):
        """
        Apply augmentation to a single image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Augmented image
        """
        if random.random() > self.p:
            return image
            
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Apply Albumentations
        augmented = self.albumentations_pipeline(image=image)
        return augmented['image']
    
    def augment_batch(self, images, labels=None):
        """
        Apply augmentation to a batch of images
        
        Args:
            images: Batch of images (numpy array)
            labels: Optional labels
            
        Returns:
            Augmented images and labels
        """
        augmented_images = []
        
        for i in range(len(images)):
            aug_img = self.augment_single_image(images[i])
            augmented_images.append(aug_img)
        
        augmented_images = np.array(augmented_images)
        
        if labels is not None:
            return augmented_images, labels
        return augmented_images
    
    def medical_specific_augmentation(self, image):
        """
        Medical-specific augmentation techniques for OCT images
        
        Args:
            image: Input OCT image
            
        Returns:
            Augmented image
        """
        if random.random() > self.p:
            return image
            
        # OCT-specific noise simulation
        if random.random() < 0.3:
            # Speckle noise (common in OCT)
            noise = np.random.normal(0, 0.05, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        # Shadow simulation (common in OCT artifacts)
        if random.random() < 0.2:
            # Create random shadow regions
            shadow_mask = np.random.random(image.shape[:2]) > 0.8
            image[shadow_mask] *= 0.3
        
        # Contrast enhancement for retinal layers
        if random.random() < 0.4:
            # Enhance contrast in specific regions
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(image.shape) == 3:
                image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                image_lab[:, :, 0] = clahe.apply(image_lab[:, :, 0])
                image = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
            else:
                image = clahe.apply(image)
        
        return image
    
    def elastic_deformation(self, image, alpha=1, sigma=50, alpha_affine=50):
        """
        Apply elastic deformation (useful for medical images)
        
        Args:
            image: Input image
            alpha: Elastic deformation parameter
            sigma: Gaussian filter parameter
            alpha_affine: Affine transformation parameter
            
        Returns:
            Deformed image
        """
        if random.random() > self.p:
            return image
            
        shape = image.shape[:2]
        shape_size = shape[:2]
        
        # Random affine transformation
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        
        # Elastic deformation
        dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        if len(image.shape) == 3:
            deformed = np.zeros_like(image)
            for i in range(image.shape[2]):
                deformed[:, :, i] = ndimage.map_coordinates(image[:, :, i], indices, order=1).reshape(shape)
        else:
            deformed = ndimage.map_coordinates(image, indices, order=1).reshape(shape)
        
        return deformed
    
    def mixup(self, images, labels, alpha=0.2):
        """
        Apply Mixup augmentation
        
        Args:
            images: Batch of images
            labels: Batch of labels
            alpha: Mixup parameter
            
        Returns:
            Mixed images and labels
        """
        if random.random() > self.p:
            return images, labels
            
        batch_size = len(images)
        weights = np.random.beta(alpha, alpha, batch_size)
        index = np.random.permutation(batch_size)
        
        x1, x2 = images, images[index]
        y1, y2 = labels, labels[index]
        
        x = np.array([w * x1[i] + (1 - w) * x2[i] for i, w in enumerate(weights)])
        y = np.array([w * y1[i] + (1 - w) * y2[i] for i, w in enumerate(weights)])
        
        return x, y
    
    def cutmix(self, images, labels, alpha=1.0):
        """
        Apply CutMix augmentation
        
        Args:
            images: Batch of images
            labels: Batch of labels
            alpha: CutMix parameter
            
        Returns:
            CutMixed images and labels
        """
        if random.random() > self.p:
            return images, labels
            
        batch_size = len(images)
        weights = np.random.beta(alpha, alpha, batch_size)
        index = np.random.permutation(batch_size)
        
        y1, y2 = labels, labels[index]
        x1, x2 = images, images[index]
        
        cut_rat = np.sqrt(1. - weights)
        cut_w = np.int32(self.image_size[1] * cut_rat)
        cut_h = np.int32(self.image_size[0] * cut_rat)
        
        x = x1.copy()
        for i in range(batch_size):
            # Random cut box
            cx = np.random.randint(self.image_size[1])
            cy = np.random.randint(self.image_size[0])
            
            bbx1 = np.clip(cx - cut_w[i] // 2, 0, self.image_size[1])
            bby1 = np.clip(cy - cut_h[i] // 2, 0, self.image_size[0])
            bbx2 = np.clip(cx + cut_w[i] // 2, 0, self.image_size[1])
            bby2 = np.clip(cy + cut_h[i] // 2, 0, self.image_size[0])
            
            x[i, bby1:bby2, bbx1:bbx2] = x2[i, bby1:bby2, bbx1:bbx2]
            
            # Adjust lambda to exactly match pixel ratio
            weights[i] = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (self.image_size[0] * self.image_size[1]))
            
        y = np.array([w * y1[i] + (1 - w) * y2[i] for i, w in enumerate(weights)])
        
        return x, y
    
    def create_augmentation_pipeline(self, augmentation_type='comprehensive'):
        """
        Create different augmentation pipelines
        
        Args:
            augmentation_type: Type of augmentation ('comprehensive', 'light', 'heavy')
            
        Returns:
            Augmentation function
        """
        if augmentation_type == 'light':
            # Light augmentation for validation
            return A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HorizontalFlip(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ], p=1.0)
        
        elif augmentation_type == 'heavy':
            # Heavy augmentation for training
            return A.Compose([
                A.Rotate(limit=20, p=0.8),
                A.HorizontalFlip(p=0.6),
                A.VerticalFlip(p=0.4),
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.7),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.4, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.7),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.7),
                A.GaussNoise(var_limit=(20.0, 80.0), p=0.7),
                A.GaussianBlur(blur_limit=(3, 9), p=0.5),
                A.CoarseDropout(max_holes=12, max_height=40, max_width=40, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ], p=1.0)
        
        else:
            # Comprehensive (default) augmentation
            return self.albumentations_pipeline


def create_tf_augmentation_pipeline(image_size=(224, 224)):
    """
    Create TensorFlow-based augmentation pipeline
    
    Args:
        image_size: Target image size
        
    Returns:
        TensorFlow augmentation pipeline
    """
    def augment_tf(image, label):
        # Convert to float32
        image = tf.cast(image, tf.float32) / 255.0
        
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random vertical flip
        image = tf.image.random_flip_up_down(image)
        
        # Random rotation
        angle = tf.random.uniform([], -0.3, 0.3)  # ±17 degrees
        image = tfa.image.rotate(image, angle)
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Random saturation and hue
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.1)
        
        # Clip values
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        # Normalize for EfficientNet
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        
        return image, label
    
    return augment_tf


def apply_augmentation_to_dataset(dataset, augmentation_pipeline, batch_size=32):
    """
    Apply augmentation to TensorFlow dataset
    
    Args:
        dataset: TensorFlow dataset
        augmentation_pipeline: Augmentation pipeline
        batch_size: Batch size
        
    Returns:
        Augmented dataset
    """
    def augment_batch(images, labels):
        # Convert to numpy for Albumentations
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        # Apply augmentation
        augmented_images = []
        for i in range(len(images_np)):
            aug_img = augmentation_pipeline(image=images_np[i])['image']
            augmented_images.append(aug_img)
        
        return tf.convert_to_tensor(augmented_images), labels
    
    # Apply augmentation to each batch
    dataset = dataset.map(
        lambda x, y: tf.py_function(
            augment_batch, 
            [x, y], 
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE)


# Example usage functions
def get_augmentation_stats(original_images, augmented_images):
    """
    Get statistics about augmentation effects
    
    Args:
        original_images: Original images
        augmented_images: Augmented images
        
    Returns:
        Dictionary with augmentation statistics
    """
    stats = {
        'mean_change': np.mean(augmented_images - original_images),
        'std_change': np.std(augmented_images - original_images),
        'min_change': np.min(augmented_images - original_images),
        'max_change': np.max(augmented_images - original_images),
    }
    return stats


def visualize_augmentations(images, augmentation_pipeline, num_samples=5):
    """
    Visualize augmentation effects
    
    Args:
        images: Original images
        augmentation_pipeline: Augmentation pipeline
        num_samples: Number of samples to visualize
        
    Returns:
        List of augmented images
    """
    augmented_samples = []
    
    for i in range(min(num_samples, len(images))):
        aug_img = augmentation_pipeline(image=images[i])['image']
        augmented_samples.append(aug_img)
    
    return augmented_samples 