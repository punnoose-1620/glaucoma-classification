#!/usr/bin/env python3
"""
GPU Test Script for TensorFlow
This script tests GPU detection and usage
"""

import tensorflow as tf
import numpy as np
import time

def test_gpu_detection():
    """Test GPU detection and configuration"""
    print("="*60)
    print("GPU DETECTION TEST")
    print("="*60)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check available devices
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print(f"Available GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    print(f"Available CPUs: {len(cpus)}")
    for i, cpu in enumerate(cpus):
        print(f"  CPU {i}: {cpu}")
    
    # Check CUDA availability
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    return len(gpus) > 0

def test_gpu_performance():
    """Test GPU performance with a simple computation"""
    print("\n" + "="*60)
    print("GPU PERFORMANCE TEST")
    print("="*60)
    
    # Create a large matrix
    size = 2000
    print(f"Creating {size}x{size} matrices...")
    
    # Test on CPU
    with tf.device('/CPU:0'):
        a_cpu = tf.random.normal([size, size])
        b_cpu = tf.random.normal([size, size])
        
        start_time = time.time()
        c_cpu = tf.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        print(f"CPU computation time: {cpu_time:.4f} seconds")
    
    # Test on GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            with tf.device('/GPU:0'):
                a_gpu = tf.random.normal([size, size])
                b_gpu = tf.random.normal([size, size])
                
                start_time = time.time()
                c_gpu = tf.matmul(a_gpu, b_gpu)
                gpu_time = time.time() - start_time
                print(f"GPU computation time: {gpu_time:.4f} seconds")
                
                if gpu_time < cpu_time:
                    speedup = cpu_time / gpu_time
                    print(f"✓ GPU is {speedup:.2f}x faster than CPU")
                else:
                    print("⚠️  GPU is slower than CPU (this might be due to small matrix size)")
                    
        except Exception as e:
            print(f"✗ GPU computation failed: {e}")
    else:
        print("No GPU available for testing")

def test_gpu_memory():
    """Test GPU memory allocation"""
    print("\n" + "="*60)
    print("GPU MEMORY TEST")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Try to allocate GPU memory
            with tf.device('/GPU:0'):
                # Allocate a large tensor
                large_tensor = tf.random.normal([1000, 1000, 100])
                print(f"✓ Successfully allocated tensor on GPU with shape: {large_tensor.shape}")
                print(f"  Tensor size: {large_tensor.shape[0] * large_tensor.shape[1] * large_tensor.shape[2] * 4 / 1024 / 1024:.2f} MB")
                
                # Perform computation
                result = tf.reduce_sum(large_tensor)
                print(f"✓ GPU computation successful: {result.numpy()}")
                
        except Exception as e:
            print(f"✗ GPU memory test failed: {e}")
    else:
        print("No GPU available for memory testing")

def main():
    """Main test function"""
    print("TensorFlow GPU Test")
    print("="*60)
    
    # Test GPU detection
    gpu_available = test_gpu_detection()
    
    # Test GPU performance
    test_gpu_performance()
    
    # Test GPU memory
    test_gpu_memory()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if gpu_available:
        print("✓ GPU is available and detected by TensorFlow")
        print("  You can now run GPU-optimized training scripts")
    else:
        print("✗ No GPU detected by TensorFlow")
        print("  Training will use CPU (slower)")
        print("  Consider installing CUDA and cuDNN for GPU support")

if __name__ == "__main__":
    main()
