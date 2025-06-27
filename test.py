import os
import json
import numpy as np
import tensorflow as tf
import h5py
import kaggle
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/testing.log'),
        logging.StreamHandler()
    ]
)

class ModelTester:
    def __init__(self):
        self.feature_types = [
            'RNFL_Input', 'Cup_to_Disc_Input', 'Rim_Input',
            'Juxtapapillary_Input', 'Sheath_Input', 'Macular_Input'
        ]
        self.classes = [
            'Normal', 'Open_angle', 'Angle_closure',
            'Normal_tension', 'Secondary'
        ]
        self.img_size = (224, 224)
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model('models/glaucoma_classifier.h5')
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
            
    def load_test_data(self):
        """Load test data from Kaggle"""
        try:
            kaggle.api.dataset_download_files(
                'aleksanderprudnik/synthetic-oct-glaucoma-dataset',
                path='.',
                unzip=True
            )
            logging.info("Test dataset downloaded successfully")
        except Exception as e:
            logging.error(f"Error downloading test dataset: {str(e)}")
            raise
            
    def prepare_test_samples(self, h5_file_path, num_samples=10):
        """Prepare random test samples"""
        with h5py.File(h5_file_path, 'r') as f:
            # Randomly select samples
            sample_indices = np.random.choice(500, num_samples, replace=False)
            
            X_test = []
            y_test = []
            
            for feature_type in self.feature_types:
                X_test.append(f['features'][feature_type][sample_indices])
                y_test.append(f['labels'][sample_indices])
                
            return np.array(X_test), np.array(y_test)
            
    def run_predictions(self, X_test):
        """Run predictions on test samples"""
        predictions = self.model.predict(X_test)
        return predictions
        
    def analyze_results(self, y_true, y_pred):
        """Analyze prediction results"""
        y_true_classes = np.argmax(y_true, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Test Set Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('visualizations/test_confusion_matrix.png')
        plt.close()
        
        # Generate classification report
        report = classification_report(
            y_true_classes,
            y_pred_classes,
            target_names=self.classes,
            output_dict=True
        )
        
        # Save report
        with open('logs/test_classification_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        return report
        
    def generate_report(self, report):
        """Generate a human-readable report"""
        print("\n=== Test Results Report ===")
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nOverall Metrics:")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Macro Avg F1-score: {report['macro avg']['f1-score']:.4f}")
        
        print("\nPer-class Metrics:")
        for class_name in self.classes:
            metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1-score']:.4f}")
            
def main():
    # Initialize tester
    tester = ModelTester()
    
    # Load model
    tester.load_model()
    
    # Load test data
    tester.load_test_data()
    
    # Prepare test samples
    X_test, y_test = tester.prepare_test_samples('glaucoma_dataset.h5')
    
    # Run predictions
    predictions = tester.run_predictions(X_test)
    
    # Analyze results
    report = tester.analyze_results(y_test, predictions)
    
    # Generate report
    tester.generate_report(report)
    
if __name__ == "__main__":
    main() 