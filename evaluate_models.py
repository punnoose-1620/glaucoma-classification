#!/usr/bin/env python3
"""
Evaluation script for the trained EfficientNet models
Fixes the label format issues and provides proper metrics
"""

import os
import json
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset_info(h5_file_path):
    """Load dataset information and process labels"""
    with h5py.File(h5_file_path, 'r') as f:
        num_samples = f['features']['cup_disc_input'].shape[0]
        labels = f['labels'][:].astype(np.int32)
        
        # Convert one-hot encoded labels to class indices
        if len(labels.shape) > 1:
            labels = np.argmax(labels, axis=1)
        
        logger.info(f"Dataset: {num_samples} samples")
        logger.info(f"Class distribution: {np.bincount(labels)}")
        
        return num_samples, labels

def load_batch_data(h5_file_path, indices, input_types=None):
    """Load batch data with proper label handling"""
    if input_types is None:
        input_types = ['cup_disc_input', 'juxta_input', 'macular_input', 
                      'rim_input', 'rnfl_input', 'sheath_input']
    
    # Sort indices for h5py compatibility
    sorted_indices = np.sort(indices)
    reverse_map = np.argsort(np.argsort(indices))
    
    with h5py.File(h5_file_path, 'r') as f:
        batch_labels = f['labels'][sorted_indices].astype(np.int32)
        
        # Convert one-hot encoded labels to class indices
        if len(batch_labels.shape) > 1:
            batch_labels = np.argmax(batch_labels, axis=1)
        
        combined_inputs = []
        for input_type in input_types:
            data = f['features'][input_type][sorted_indices].astype(np.float32)
            combined_inputs.append(data)
        
        combined_inputs = np.mean(combined_inputs, axis=0)
        combined_inputs = (combined_inputs - combined_inputs.min()) / (combined_inputs.max() - combined_inputs.min() + 1e-8)
        
        if len(combined_inputs.shape) == 3:
            combined_inputs = np.stack([combined_inputs] * 3, axis=-1)
        
        combined_inputs = tf.keras.applications.efficientnet.preprocess_input(combined_inputs)
        
        # Restore original order
        combined_inputs = combined_inputs[reverse_map]
        batch_labels = batch_labels[reverse_map]
        
        return combined_inputs, batch_labels

def evaluate_model(model_path, val_indices, h5_file_path):
    """Evaluate a single model"""
    logger.info(f"Loading model: {model_path}")
    
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None
    
    # Collect predictions and true labels
    predictions = []
    true_labels = []
    
    # Process in batches
    batch_size = 16
    for i in range(0, len(val_indices), batch_size):
        batch_indices = val_indices[i:i+batch_size]
        batch_images, batch_labels = load_batch_data(h5_file_path, batch_indices)
        
        # Get predictions
        batch_predictions = model.predict(batch_images, verbose=0)
        batch_pred_indices = np.argmax(batch_predictions, axis=1)
        
        predictions.extend(batch_pred_indices)
        true_labels.extend(batch_labels)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Classification report
    class_names = ['Normal', 'Open-angle', 'Angle-closure', 'Normal-tension', 'Secondary']
    report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist()
    }
    
    logger.info(f"Model evaluation completed:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return results

def main():
    """Main evaluation function"""
    logger.info("Starting EfficientNet Model Evaluation")
    
    # Configuration
    h5_file_path = 'synthetic_glaucoma_data.h5'
    models_dir = 'models'
    
    if not os.path.exists(h5_file_path):
        logger.error(f"Dataset not found: {h5_file_path}")
        return
    
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return
    
    # Load dataset info
    num_samples, labels = load_dataset_info(h5_file_path)
    
    # Initialize cross-validation to get the same splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Find trained models
    model_files = [f for f in os.listdir(models_dir) if f.startswith('efficientnet_fold_') and f.endswith('.h5')]
    model_files.sort()
    
    if not model_files:
        logger.error("No trained models found")
        return
    
    logger.info(f"Found {len(model_files)} trained models")
    
    # Evaluate each model
    all_results = []
    
    for fold, (train_indices, val_indices) in enumerate(skf.split(np.arange(num_samples), labels)):
        model_path = os.path.join(models_dir, f'efficientnet_fold_{fold + 1}.h5')
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            continue
        
        logger.info(f"Evaluating Fold {fold + 1}")
        results = evaluate_model(model_path, val_indices, h5_file_path)
        
        if results:
            results['fold'] = fold + 1
            all_results.append(results)
    
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
        results_file = f'logs/efficientnet_evaluation_{timestamp}.json'
        
        os.makedirs('logs', exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'evaluation_timestamp': timestamp,
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
        
        # Print detailed results for each fold
        print("\n" + "="*60)
        print("DETAILED RESULTS BY FOLD")
        print("="*60)
        
        for result in all_results:
            print(f"\nFold {result['fold']}:")
            print(f"  Accuracy:  {result['accuracy']:.4f}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall:    {result['recall']:.4f}")
            print(f"  F1 Score:  {result['f1']:.4f}")
        
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        print(f"Average Accuracy:  {avg_accuracy:.4f} ± {np.std([r['accuracy'] for r in all_results]):.4f}")
        print(f"Average Precision: {avg_precision:.4f} ± {np.std([r['precision'] for r in all_results]):.4f}")
        print(f"Average Recall:    {avg_recall:.4f} ± {np.std([r['recall'] for r in all_results]):.4f}")
        print(f"Average F1 Score:  {avg_f1:.4f} ± {np.std([r['f1'] for r in all_results]):.4f}")
        
    else:
        logger.error("No successful evaluations completed")

if __name__ == "__main__":
    main()
