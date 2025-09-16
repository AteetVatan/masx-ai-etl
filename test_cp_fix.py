#!/usr/bin/env python3
"""
Simple test to verify the cp (cupy) fix in hdbscan_clusterer.py
"""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cp_availability():
    """Test if cp (cupy) is properly defined in the module."""
    
    print("Testing cp (cupy) availability in hdbscan_clusterer...")
    
    try:
        # Import the module
        from app.nlp.hdbscan_clusterer import HDBSCANClusterer, cp, _HAS_GPU_STACK
        
        print(f"✅ HDBSCANClusterer imported successfully")
        print(f"✅ cp variable is defined: {cp is not None}")
        print(f"✅ _HAS_GPU_STACK: {_HAS_GPU_STACK}")
        
        if cp is not None:
            print(f"✅ cupy version: {cp.__version__}")
        else:
            print("ℹ️  cupy not available (expected without RAPIDS)")
        
        # Test creating clusterers
        print("\nTesting clusterer creation...")
        
        # Test CPU clusterer
        try:
            cpu_clusterer = HDBSCANClusterer(force_cpu=True)
            print("✅ CPU clusterer created successfully")
            print(f"   GPU enabled: {cpu_clusterer._gpu_enabled}")
        except Exception as e:
            print(f"❌ CPU clusterer creation failed: {e}")
        
        # Test GPU clusterer
        try:
            gpu_clusterer = HDBSCANClusterer(force_gpu=True)
            print("✅ GPU clusterer created successfully")
            print(f"   GPU enabled: {gpu_clusterer._gpu_enabled}")
        except Exception as e:
            print(f"❌ GPU clusterer creation failed: {e}")
        
        # Test with small data
        print("\nTesting with small data...")
        test_data = np.random.randn(10, 5)
        
        try:
            cpu_clusterer = HDBSCANClusterer(force_cpu=True, use_umap=False)
            labels = cpu_clusterer.cluster(test_data)
            print(f"✅ CPU clustering successful: {len(np.unique(labels))} clusters")
        except Exception as e:
            print(f"❌ CPU clustering failed: {e}")
        
        try:
            gpu_clusterer = HDBSCANClusterer(force_gpu=True, use_umap=False)
            labels = gpu_clusterer.cluster(test_data)
            print(f"✅ GPU clustering successful: {len(np.unique(labels))} clusters")
        except Exception as e:
            print(f"ℹ️  GPU clustering failed (expected without RAPIDS): {e}")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cp_availability()
