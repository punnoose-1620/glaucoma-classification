# Glaucoma Classification Project

This project implements a multi-class glaucoma classification system using multiple ophthalmic imaging features. The system uses a deep learning approach to classify glaucoma into five categories: Normal, Open-angle, Angle-closure, Normal-tension, and Secondary glaucoma.

## Dataset

The project uses a synthetic OCT glaucoma dataset from Kaggle containing 3000 grayscale images (224x224 pixels) distributed across 6 feature types:
- RNFL Input
- Cup-to-Disc Input
- Rim Input
- Juxtapapillary Input
- Sheath Input
- Macular Input

Each feature type contains 500 images, and the labels are one-hot encoded for the 5 glaucoma classes.

## Project Structure

```
.
├── glaucoma_classification.py      # Original CNN training script
├── efficientnet_glaucoma_training.py  # EfficientNet training script
├── data_augmentation.py           # Medical image augmentation module
├── memory_utils.py                # Memory management utilities
├── test.py                        # Testing script
├── requirements.txt               # Project dependencies
├── setup.sh                       # Setup script
├── visualizations/                # Generated plots and figures
├── logs/                          # Training logs and metrics
├── checkpoints/                   # Model checkpoints
└── models/                        # Saved model files
```

## Model Comparison: CNN vs EfficientNet

### **Previous CNN Model Architecture**

#### **Architecture Details:**
- **Type**: Custom multi-input CNN with residual connections
- **Input Processing**: 6 separate CNN branches for each feature type
- **Convolutional Layers**: 
  - 3 blocks with increasing filter sizes (64→128→256)
  - Each block: 2 Conv2D + BatchNorm + ReLU + MaxPooling
- **Feature Fusion**: Concatenation of all branch outputs
- **Dense Layers**: 512 → 512 (with residual) → 256 → 5 (output)
- **Parameters**: ~4.8M parameters
- **Optimizer**: AdamW with weight decay (0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 16
- **Learning Rate**: 0.0001

#### **Training Features:**
- Early stopping with patience=15
- Learning rate scheduling (ReduceLROnPlateau)
- Model checkpointing
- Class weight balancing
- Basic data augmentation

#### **Performance Results:**
- **Average Accuracy**: 19.2% (5-fold CV)
- **Average Precision**: 4.8%
- **Average Recall**: 19.2%
- **Average F1-Score**: 7.5%
- **Training Time**: ~2-3 hours
- **Memory Usage**: ~6-8GB

### **Current EfficientNet Model Architecture**

#### **Architecture Details:**
- **Type**: EfficientNetB0 pre-trained + custom head
- **Base Model**: EfficientNetB0 (ImageNet weights)
- **Input Processing**: Single RGB input (224x224x3)
- **Transfer Learning**: Two-phase training (frozen + fine-tuned)
- **Custom Head**: 
  - GlobalAveragePooling2D
  - BatchNormalization + Dropout(0.5)
  - Dense(512) + BatchNorm + Dropout(0.5)
  - Dense(256) + BatchNorm + Dropout(0.5)
  - Dense(5, softmax)
- **Parameters**: ~4.8M parameters (same as CNN)
- **Optimizer**: Adam with Cosine Annealing
- **Loss Function**: Focal Loss (α=1, γ=2)
- **Batch Size**: 16
- **Learning Rate**: 0.001 (initial)

#### **Advanced Training Features:**
- **Focal Loss**: Handles class imbalance better than crossentropy
- **Cosine Annealing**: Smooth learning rate decay
- **Memory Management**: 8GB RAM limit with garbage collection
- **Advanced Augmentation**: Medical-specific augmentations
- **Two-Phase Training**: Frozen base + fine-tuning
- **Mixed Precision**: FP16 training for efficiency
- **Comprehensive Logging**: TensorBoard integration

#### **Performance Results:**
- **Average Accuracy**: 20.2% ± 1.83%
- **Average Precision**: 4.11% ± 0.72%
- **Average Recall**: 20.2% ± 1.83%
- **Average F1-Score**: 6.85% ± 1.21%
- **Training Time**: ~4-5 hours (longer due to fine-tuning)
- **Memory Usage**: ~8-8.5GB (higher due to pre-trained model)

### **Detailed Performance Analysis**

#### **Cross-Validation Results Comparison:**

| Metric | CNN Model | EfficientNet Model | Difference |
|--------|-----------|-------------------|------------|
| **Fold 1 Accuracy** | 19% | 22% | +3% |
| **Fold 2 Accuracy** | 22% | 18% | -4% |
| **Fold 3 Accuracy** | 19% | 22% | +3% |
| **Fold 4 Accuracy** | 20% | 20% | 0% |
| **Fold 5 Accuracy** | 15% | 19% | +4% |
| **Average Accuracy** | 19.2% | 20.2% ± 1.83% | +1% |
| **Average Precision** | 4.8% | 4.11% ± 0.72% | -0.69% |
| **Average Recall** | 19.2% | 20.2% ± 1.83% | +1% |
| **Average F1-Score** | 7.5% | 6.85% ± 1.21% | -0.65% |

#### **Class-Specific Performance:**

| Class | CNN Precision | CNN Recall | EfficientNet Precision | EfficientNet Recall |
|-------|---------------|------------|------------------------|---------------------|
| **Normal** | 0% | 0% | 0% | 0% |
| **Open_angle** | 22.2% | 47.1% | 22.2% | 47.1% |
| **Angle_closure** | 23.1% | 56.3% | 23.1% | 56.3% |
| **Normal_tension** | 0% | 0% | 0% | 0% |
| **Secondary** | 0% | 0% | 0% | 0% |

### **Key Findings and Analysis**

#### **Performance Similarity:**
Both models achieved **identical performance metrics**, suggesting that:
1. The dataset may have fundamental issues (class imbalance, data quality)
2. The problem complexity requires different approaches
3. Both architectures are equally limited by the available data

#### **Architecture Comparison:**

| Aspect | CNN Model | EfficientNet Model | Advantage |
|--------|-----------|-------------------|-----------|
| **Complexity** | Custom architecture | Pre-trained + fine-tuning | EfficientNet |
| **Feature Processing** | Multi-input branches | Single input | CNN (multi-modal) |
| **Transfer Learning** | None | ImageNet weights | EfficientNet |
| **Loss Function** | Standard crossentropy | Focal Loss | EfficientNet |
| **Regularization** | Dropout + BatchNorm | Dropout + BatchNorm | Equal |
| **Memory Efficiency** | Moderate | Higher | CNN |
| **Training Speed** | Faster | Slower | CNN |

#### **Identified Issues:**

1. **Class Imbalance**: Both models fail on 3 out of 5 classes (0% precision/recall)
2. **Data Quality**: The synthetic dataset may not represent real clinical scenarios
3. **Model Collapse**: Models predict only 1-2 classes for all samples
4. **Memory Constraints**: Both models exceed memory limits during training

#### **Recommendations for Improvement:**

1. **Data Quality**:
   - Use real clinical OCT images instead of synthetic data
   - Ensure proper class balance in training data
   - Validate image quality and labeling accuracy

2. **Architecture Improvements**:
   - Implement attention mechanisms for multi-modal fusion
   - Use ensemble methods combining multiple architectures
   - Explore Vision Transformers (ViT) for better feature learning

3. **Training Enhancements**:
   - Implement stronger data augmentation
   - Use curriculum learning approaches
   - Explore self-supervised pre-training on medical images

4. **Evaluation**:
   - Use additional metrics (AUC-ROC, Cohen's Kappa)
   - Implement cross-dataset validation
   - Add clinical validation with expert ophthalmologists

### **Conclusion**

The EfficientNet model achieved **slightly better accuracy** (20.2% vs 19.2%) compared to the custom CNN model, but both models show similar overall performance limitations. The EfficientNet model offers more advanced training features (Focal Loss, transfer learning, better augmentation) and demonstrates marginal improvements in accuracy and recall, but the fundamental dataset constraints limit significant performance gains.

**Key Performance Insights:**
- **Accuracy**: EfficientNet shows +1% improvement (20.2% vs 19.2%)
- **Precision**: CNN slightly better (4.8% vs 4.11%)
- **Recall**: EfficientNet shows +1% improvement (20.2% vs 19.2%)
- **F1-Score**: CNN slightly better (7.5% vs 6.85%)

**Next Steps:**
1. Acquire real clinical OCT data with proper class balance
2. Implement ensemble methods combining multiple architectures
3. Add clinical validation and expert consultation
4. Explore domain-specific pre-training on medical imaging datasets

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd glaucoma-classification
```

2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/settings/account
   - Download your API credentials (kaggle.json)
   - Place the file in ~/.kaggle/
   - Set appropriate permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. Run the setup script:
```bash
./setup.sh
```

4. Activate the virtual environment:
```bash
source venv/bin/activate
```

## Usage

### Training the Original CNN Model

To train the original multi-input CNN model, run:
```bash
python glaucoma_classification.py
```

### Training the EfficientNet Model

To train the EfficientNet model with advanced features, run:
```bash
python efficientnet_glaucoma_training.py
```

### Testing the Model

To test any trained model on new samples, run:
```bash
python test.py
```

## Model Architecture

### Original CNN Model
The original model uses a multi-input CNN architecture that:
1. Processes each feature type through separate CNN branches
2. Concatenates the features
3. Uses dense layers with residual connections for final classification

### EfficientNet Model
The EfficientNet model uses:
1. Pre-trained EfficientNetB0 as feature extractor
2. Custom classification head with dropout regularization
3. Two-phase training (frozen base + fine-tuning)
4. Focal Loss for handling class imbalance

Key features:
- Early stopping to prevent overfitting
- Learning rate scheduling (Cosine Annealing)
- Model checkpointing
- Comprehensive logging and visualization
- Memory-efficient training

## Outputs

The project generates several outputs:

### Visualizations
- Sample images from each feature type
- Confusion matrices
- Training curves
- Feature importance visualizations

### Logs
- Training history
- Classification reports
- Performance metrics
- Cross-validation results

### Models
- Saved model weights
- Model architecture
- ONNX format for deployment

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- CUDA-compatible GPU (recommended)
- See requirements.txt for full list of dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided by Aleksander Prudnik on Kaggle
- Built with TensorFlow and other open-source libraries 