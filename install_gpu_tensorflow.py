#!/usr/bin/env python3
"""
Install TensorFlow with GPU support for NVIDIA RTX 4060
"""

import subprocess
import sys
import os

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected:")
            print(result.stdout)
            return True
        else:
            print("✗ NVIDIA GPU not detected")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found. Please install NVIDIA drivers.")
        return False

def install_gpu_tensorflow():
    """Install TensorFlow with GPU support"""
    print("="*60)
    print("INSTALLING TENSORFLOW WITH GPU SUPPORT")
    print("="*60)
    
    # Check if NVIDIA GPU is available
    if not check_nvidia_gpu():
        print("Cannot proceed without NVIDIA GPU detection.")
        return False
    
    print("\nInstalling TensorFlow with GPU support...")
    
    # Uninstall current TensorFlow
    print("1. Uninstalling current TensorFlow...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "tensorflow", "-y"])
    
    # Install TensorFlow with GPU support
    print("2. Installing TensorFlow with GPU support...")
    
    # Try different TensorFlow versions that support GPU
    tf_versions = [
        "tensorflow==2.15.0",
        "tensorflow==2.14.0", 
        "tensorflow==2.13.0",
        "tensorflow"  # Latest version
    ]
    
    for version in tf_versions:
        print(f"   Trying {version}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", version
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully installed {version}")
                return True
            else:
                print(f"✗ Failed to install {version}")
                print(f"   Error: {result.stderr}")
                
        except Exception as e:
            print(f"✗ Error installing {version}: {e}")
    
    print("✗ Failed to install any TensorFlow version with GPU support")
    return False

def test_gpu_installation():
    """Test if GPU is now detected"""
    print("\n" + "="*60)
    print("TESTING GPU INSTALLATION")
    print("="*60)
    
    test_code = """
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")
print(f"Available CPUs: {tf.config.list_physical_devices('CPU')}")

# Test GPU computation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✓ GPU detected! Testing computation...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print(f"✓ GPU computation successful: {c.shape}")
else:
    print("✗ No GPU detected")
"""
    
    try:
        result = subprocess.run([
            sys.executable, "-c", test_code
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        return "GPU" in result.stdout and "✓ GPU detected" in result.stdout
        
    except Exception as e:
        print(f"Error testing GPU: {e}")
        return False

def main():
    """Main installation function"""
    print("TensorFlow GPU Installation for NVIDIA RTX 4060")
    print("="*60)
    
    # Check current setup
    print("Current setup:")
    check_nvidia_gpu()
    
    # Install TensorFlow with GPU support
    if install_gpu_tensorflow():
        print("\n✓ TensorFlow with GPU support installed successfully!")
        
        # Test the installation
        if test_gpu_installation():
            print("\n" + "="*60)
            print("SUCCESS! GPU is now available for TensorFlow")
            print("="*60)
            print("You can now run:")
            print("  python gpu_optimized_training.py")
            print("  python test_gpu.py")
        else:
            print("\n⚠️  GPU installation may not be working correctly")
            print("   You may need to install CUDA and cuDNN manually")
    else:
        print("\n✗ Failed to install TensorFlow with GPU support")
        print("   You may need to install CUDA and cuDNN manually")

if __name__ == "__main__":
    main()
