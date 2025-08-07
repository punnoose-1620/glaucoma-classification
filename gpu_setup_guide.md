# GPU Setup Guide for TensorFlow on Windows

## Current Status
Your system has an NVIDIA GeForce RTX 4060 with CUDA 12.9, but TensorFlow is not detecting the GPU because it's not built with CUDA support.

## Solution Options

### Option 1: Use Current TensorFlow with CPU (Recommended for now)
The current TensorFlow installation will use CPU for training. While slower than GPU, it will still work for your glaucoma classification project.

### Option 2: Install TensorFlow with GPU Support (Advanced)
To enable GPU support, you would need to:

1. **Install CUDA Toolkit 11.8** (compatible with TensorFlow 2.15)
   - Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Choose Windows x86_64

2. **Install cuDNN 8.6** (compatible with CUDA 11.8)
   - Download from: https://developer.nvidia.com/cudnn
   - Requires NVIDIA developer account

3. **Install TensorFlow 2.15.0 with GPU support**
   ```bash
   pip uninstall tensorflow
   pip install tensorflow==2.15.0
   ```

4. **Set environment variables**
   ```bash
   set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
   set PATH=%CUDA_PATH%\bin;%PATH%
   ```

## Current Training Scripts

### GPU-Optimized Training Script
The `gpu_optimized_training.py` script is configured to:
- Automatically detect GPU if available
- Fall back to CPU if GPU is not available
- Use memory-efficient data loading
- Optimize for the best available hardware

### Running the Training

1. **Test GPU detection:**
   ```bash
   python test_gpu.py
   ```

2. **Run GPU-optimized training:**
   ```bash
   python gpu_optimized_training.py
   ```

3. **Monitor training progress:**
   - Check the logs in the `logs/` directory
   - View visualizations in the `visualizations/` directory
   - Monitor GPU usage with `nvidia-smi` (if GPU is available)

## Performance Expectations

### With CPU (Current Setup)
- Training time: 2-4 hours for full dataset
- Memory usage: ~8GB RAM
- Suitable for development and testing

### With GPU (If properly configured)
- Training time: 30-60 minutes for full dataset
- Memory usage: ~4GB GPU memory + 4GB RAM
- Much faster for production training

## Troubleshooting

### If GPU is not detected:
1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Verify TensorFlow GPU support: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### If training is slow:
1. Reduce batch size in the training script
2. Use fewer epochs for testing
3. Consider using a subset of the data for initial testing

## Next Steps

1. **Start with CPU training** using `gpu_optimized_training.py`
2. **Monitor the training progress** and check for any errors
3. **If you want GPU acceleration**, follow the GPU setup instructions above
4. **For production use**, consider setting up proper GPU support

## Files Created

- `gpu_optimized_training.py`: Main training script with GPU optimization
- `test_gpu.py`: GPU detection and performance test script
- `gpu_setup_guide.md`: This setup guide

The training script will automatically use the best available hardware (GPU if available, CPU otherwise) and provide detailed logging of the training process.
