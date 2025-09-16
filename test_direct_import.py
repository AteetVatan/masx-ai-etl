#!/usr/bin/env python3
"""
Direct test of hdbscan_clusterer.py without importing the full app module.
"""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_direct_import():
    """Test direct import of hdbscan_clusterer module."""
    
    print("Testing direct import of hdbscan_clusterer...")
    
    try:
        # Import directly from the file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hdbscan_clusterer", 
            "app/nlp/hdbscan_clusterer.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print("✅ hdbscan_clusterer module loaded successfully")
        
        # Check if cp is defined
        cp = getattr(module, 'cp', None)
        _HAS_GPU_STACK = getattr(module, '_HAS_GPU_STACK', False)
        
        print(f"✅ cp variable is defined: {cp is not None}")
        print(f"✅ _HAS_GPU_STACK: {_HAS_GPU_STACK}")
        
        if cp is not None:
            print(f"✅ cupy version: {cp.__version__}")
        else:
            print("ℹ️  cupy not available (expected without RAPIDS)")
        
        # Test the HDBSCANClusterer class
        HDBSCANClusterer = getattr(module, 'HDBSCANClusterer', None)
        
        if HDBSCANClusterer is not None:
            print("✅ HDBSCANClusterer class found")
            
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
                print(f"   Labels: {np.unique(labels)}")
            except Exception as e:
                print(f"❌ CPU clustering failed: {e}")
                import traceback
                traceback.print_exc()
            
            try:
                gpu_clusterer = HDBSCANClusterer(force_gpu=True, use_umap=False)
                labels = gpu_clusterer.cluster(test_data)
                print(f"✅ GPU clustering successful: {len(np.unique(labels))} clusters")
                print(f"   Labels: {np.unique(labels)}")
            except Exception as e:
                print(f"ℹ️  GPU clustering failed (expected without RAPIDS): {e}")
        
        else:
            print("❌ HDBSCANClusterer class not found")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_import()
