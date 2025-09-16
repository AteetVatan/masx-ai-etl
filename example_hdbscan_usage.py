#!/usr/bin/env python3
"""
Example usage of HDBSCANClusterer with forced CPU and GPU scenarios.
"""

import numpy as np
import asyncio
from app.nlp.hdbscan_clusterer import HDBSCANClusterer

def main():
    """Example usage of HDBSCANClusterer."""
    
    # Generate some test data
    print("Generating test data...")
    np.random.seed(42)
    
    # Create 3 distinct clusters
    cluster1 = np.random.normal([1, 1], 0.3, (50, 2))
    cluster2 = np.random.normal([-1, -1], 0.3, (50, 2))
    cluster3 = np.random.normal([0, 2], 0.3, (50, 2))
    
    # Combine and shuffle
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    np.random.shuffle(embeddings)
    
    print(f"Test data shape: {embeddings.shape}")
    
    # Example 1: Force CPU mode
    print("\n" + "="*60)
    print("Example 1: Force CPU Mode")
    print("="*60)
    
    cpu_clusterer = HDBSCANClusterer(
        force_cpu=True,           # Force CPU mode
        use_umap=True,            # Use UMAP for dimensionality reduction
        min_cluster_size=5,       # Minimum cluster size
        random_state=42           # For reproducibility
    )
    
    print(f"GPU enabled: {cpu_clusterer._gpu_enabled}")
    
    # Test both sync and async methods
    labels_sync = cpu_clusterer.cluster(embeddings)
    print(f"Sync clustering result: {len(np.unique(labels_sync))} clusters")
    
    # Example 2: Force GPU mode (will fallback to CPU if no GPU)
    print("\n" + "="*60)
    print("Example 2: Force GPU Mode")
    print("="*60)
    
    gpu_clusterer = HDBSCANClusterer(
        force_gpu=True,           # Force GPU mode
        use_umap=False,           # Skip UMAP for simplicity
        min_cluster_size=5,       # Minimum cluster size
        random_state=42           # For reproducibility
    )
    
    print(f"GPU enabled: {gpu_clusterer._gpu_enabled}")
    
    # Test async method
    async def test_async():
        labels_async = await gpu_clusterer.cluster_async(embeddings)
        print(f"Async clustering result: {len(np.unique(labels_async))} clusters")
        return labels_async
    
    labels_async = asyncio.run(test_async())
    
    # Example 3: Auto mode (default behavior)
    print("\n" + "="*60)
    print("Example 3: Auto Mode (Default)")
    print("="*60)
    
    auto_clusterer = HDBSCANClusterer(
        # No force_cpu or force_gpu - uses automatic detection
        use_umap=True,
        min_cluster_size=5,
        random_state=42
    )
    
    print(f"GPU enabled: {auto_clusterer._gpu_enabled}")
    
    labels_auto = auto_clusterer.cluster(embeddings)
    print(f"Auto clustering result: {len(np.unique(labels_auto))} clusters")
    
    # Compare results
    print("\n" + "="*60)
    print("Results Comparison")
    print("="*60)
    
    print(f"CPU sync labels:     {np.unique(labels_sync)}")
    print(f"GPU async labels:    {np.unique(labels_async)}")
    print(f"Auto sync labels:    {np.unique(labels_auto)}")
    
    # Check if results are similar (allowing for different cluster IDs)
    cpu_clusters = len(np.unique(labels_sync))
    gpu_clusters = len(np.unique(labels_async))
    auto_clusters = len(np.unique(labels_auto))
    
    print(f"\nNumber of clusters found:")
    print(f"  CPU:  {cpu_clusters}")
    print(f"  GPU:  {gpu_clusters}")
    print(f"  Auto: {auto_clusters}")

if __name__ == "__main__":
    main()
