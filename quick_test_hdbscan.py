#!/usr/bin/env python3
"""
Quick test script for HDBSCANClusterer CPU and GPU scenarios.
"""

import numpy as np
import asyncio
from app.nlp.hdbscan_clusterer import HDBSCANClusterer

async def quick_test():
    """Quick test of both CPU and GPU scenarios."""
    
    # Generate small test data
    print("Generating test data...")
    np.random.seed(42)
    embeddings = np.random.randn(100, 32)  # 100 samples, 32 features
    
    print(f"Test data shape: {embeddings.shape}")
    
    # Test CPU scenario
    print("\n" + "="*50)
    print("Testing CPU Scenario (Forced)")
    print("="*50)
    
    try:
        cpu_clusterer = HDBSCANClusterer(
            force_cpu=True,
            use_umap=True,
            min_cluster_size=3,
            random_state=42
        )
        
        print("CPU clusterer created successfully")
        print(f"GPU enabled: {cpu_clusterer._gpu_enabled}")
        
        cpu_labels = await cpu_clusterer.cluster_async(embeddings)
        print(f"CPU clustering completed!")
        print(f"Number of clusters: {len(np.unique(cpu_labels))}")
        print(f"Labels: {np.unique(cpu_labels)}")
        
    except Exception as e:
        print(f"CPU test failed: {e}")
    
    # Test GPU scenario
    print("\n" + "="*50)
    print("Testing GPU Scenario (Forced)")
    print("="*50)
    
    try:
        gpu_clusterer = HDBSCANClusterer(
            force_gpu=True,
            use_umap=True,
            min_cluster_size=3,
            random_state=42
        )
        
        print("GPU clusterer created successfully")
        print(f"GPU enabled: {gpu_clusterer._gpu_enabled}")
        
        gpu_labels = await gpu_clusterer.cluster_async(embeddings)
        print(f"GPU clustering completed!")
        print(f"Number of clusters: {len(np.unique(gpu_labels))}")
        print(f"Labels: {np.unique(gpu_labels)}")
        
    except Exception as e:
        print(f"GPU test failed: {e}")
        print("This is expected if you don't have RAPIDS/cuML installed")
    
    # Test auto mode
    print("\n" + "="*50)
    print("Testing Auto Mode (Default)")
    print("="*50)
    
    try:
        auto_clusterer = HDBSCANClusterer(
            use_umap=True,
            min_cluster_size=3,
            random_state=42
        )
        
        print("Auto clusterer created successfully")
        print(f"GPU enabled: {auto_clusterer._gpu_enabled}")
        
        auto_labels = await auto_clusterer.cluster_async(embeddings)
        print(f"Auto clustering completed!")
        print(f"Number of clusters: {len(np.unique(auto_labels))}")
        print(f"Labels: {np.unique(auto_labels)}")
        
    except Exception as e:
        print(f"Auto test failed: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())
