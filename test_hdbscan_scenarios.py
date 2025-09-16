#!/usr/bin/env python3
"""
Test script for HDBSCANClusterer CPU and GPU scenarios.

This script allows you to test both CPU and GPU clustering paths
even without a physical GPU by using the force_cpu and force_gpu parameters.
"""

import numpy as np
import asyncio
import time
from app.nlp.hdbscan_clusterer import HDBSCANClusterer
from app.config import get_service_logger

logger = get_service_logger("HDBSCANTest")

def generate_test_data(n_samples: int = 1000, n_features: int = 128) -> np.ndarray:
    """Generate synthetic test data for clustering."""
    logger.info(f"Generating {n_samples} samples with {n_features} features...")
    
    # Create 3 distinct clusters with some noise
    np.random.seed(42)
    
    # Cluster 1: centered around [1, 1, 1, ...]
    cluster1 = np.random.normal(1, 0.3, (n_samples // 3, n_features))
    
    # Cluster 2: centered around [-1, -1, -1, ...]
    cluster2 = np.random.normal(-1, 0.3, (n_samples // 3, n_features))
    
    # Cluster 3: centered around [0, 2, 0, ...]
    cluster3 = np.random.normal(0, 0.3, (n_samples - 2 * (n_samples // 3), n_features))
    cluster3[:, 1] += 2  # Shift second dimension
    
    # Combine all clusters
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    
    # Shuffle the data
    np.random.shuffle(embeddings)
    
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings

async def test_clustering_scenario(clusterer: HDBSCANClusterer, embeddings: np.ndarray, scenario_name: str):
    """Test a specific clustering scenario."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {scenario_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Test async clustering
        labels = await clusterer.cluster_async(embeddings)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(np.array(labels) == -1)
        
        logger.info(f"✅ {scenario_name} completed successfully!")
        logger.info(f"   Duration: {duration:.2f} seconds")
        logger.info(f"   Number of clusters found: {n_clusters}")
        logger.info(f"   Number of noise points: {n_noise}")
        logger.info(f"   Total points: {len(labels)}")
        logger.info(f"   Unique labels: {unique_labels}")
        
        # Test sync clustering for comparison
        # start_sync = time.time()
        # labels_sync = clusterer.cluster(embeddings)
        # end_sync = time.time()
        
        # logger.info(f"   Sync duration: {end_sync - start_sync:.2f} seconds")
        # logger.info(f"   Async vs Sync results match: {np.array_equal(labels, labels_sync)}")
        
        return {
            'scenario': scenario_name,
            'duration': duration,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'success': True
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error(f"❌ {scenario_name} failed after {duration:.2f} seconds")
        logger.error(f"   Error: {str(e)}")
        
        return {
            'scenario': scenario_name,
            'duration': duration,
            'success': False,
            'error': str(e)
        }

async def main():
    """Main test function."""
    logger.info("Starting HDBSCANClusterer CPU/GPU Testing")
    logger.info("=" * 60)
    
    # Generate test data
    embeddings = generate_test_data(n_samples=500, n_features=64)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'GPU Mode (Forced)',
            'params': {
                'force_gpu': True,
                'use_umap': True,
                'min_cluster_size': 5,
                'random_state': 42
            }
        },
        {
            'name': 'GPU Mode (No UMAP)',
            'params': {
                'force_gpu': True,
                'use_umap': False,
                'min_cluster_size': 5,
                'random_state': 42
            }
        },
        {
            'name': 'CPU Mode (Forced)',
            'params': {
                'force_cpu': True,
                'use_umap': True,
                'min_cluster_size': 5,
                'random_state': 42
            }
        },
        {
            'name': 'CPU Mode (No UMAP)',
            'params': {
                'force_cpu': True,
                'use_umap': False,
                'min_cluster_size': 5,
                'random_state': 42
            }
        },
        {
            'name': 'Auto Mode (Default)',
            'params': {
                'use_umap': True,
                'min_cluster_size': 5,
                'random_state': 42
            }
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        logger.info(f"\nCreating clusterer for: {scenario['name']}")
        
        try:
            clusterer = HDBSCANClusterer(**scenario['params'])
            result = await test_clustering_scenario(clusterer, embeddings, scenario['name'])
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to create clusterer for {scenario['name']}: {e}")
            results.append({
                'scenario': scenario['name'],
                'success': False,
                'error': f"Clusterer creation failed: {str(e)}"
            })
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    logger.info(f"✅ Successful tests: {len(successful_tests)}")
    logger.info(f"❌ Failed tests: {len(failed_tests)}")
    
    if successful_tests:
        logger.info("\nSuccessful scenarios:")
        for result in successful_tests:
            logger.info(f"  - {result['scenario']}: {result['duration']:.2f}s, {result['n_clusters']} clusters")
    
    if failed_tests:
        logger.info("\nFailed scenarios:")
        for result in failed_tests:
            logger.info(f"  - {result['scenario']}: {result.get('error', 'Unknown error')}")
    
    # Performance comparison
    if len(successful_tests) > 1:
        logger.info("\nPerformance comparison:")
        cpu_tests = [r for r in successful_tests if 'CPU' in r['scenario']]
        gpu_tests = [r for r in successful_tests if 'GPU' in r['scenario']]
        
        if cpu_tests and gpu_tests:
            avg_cpu_time = np.mean([r['duration'] for r in cpu_tests])
            avg_gpu_time = np.mean([r['duration'] for r in gpu_tests])
            speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
            
            logger.info(f"  Average CPU time: {avg_cpu_time:.2f}s")
            logger.info(f"  Average GPU time: {avg_gpu_time:.2f}s")
            logger.info(f"  GPU speedup: {speedup:.2f}x")

if __name__ == "__main__":
    asyncio.run(main())
