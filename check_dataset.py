#!/usr/bin/env python3
"""
Check dataset structure and feature shapes
"""

import h5py
import numpy as np

def check_dataset():
    """Check the dataset structure"""
    with h5py.File('synthetic_glaucoma_data.h5', 'r') as f:
        print("Dataset structure:")
        print(f"Keys: {list(f.keys())}")
        
        print("\nLabels:")
        print(f"  Shape: {f['labels'].shape}")
        print(f"  Type: {f['labels'].dtype}")
        
        print("\nFeatures:")
        for key in f['features'].keys():
            print(f"  {key}: {f['features'][key].shape}")
        
        print("\nMetadata:")
        print(f"  Keys: {list(f['metadata'].keys())}")
        
        # Check a sample
        print("\nSample data:")
        sample_idx = 0
        print(f"  Labels[{sample_idx}]: {f['labels'][sample_idx]}")
        for key in f['features'].keys():
            print(f"  {key}[{sample_idx}]: shape {f['features'][key][sample_idx].shape}, min={f['features'][key][sample_idx].min():.3f}, max={f['features'][key][sample_idx].max():.3f}")

if __name__ == "__main__":
    check_dataset()
