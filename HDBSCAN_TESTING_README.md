# HDBSCANClusterer Testing Guide

This guide explains how to test both CPU and GPU scenarios in the HDBSCANClusterer, even without a physical GPU.

## Overview

The HDBSCANClusterer now supports forced CPU and GPU modes for testing purposes through the following parameters:

- `force_cpu=True`: Forces CPU mode regardless of GPU availability
- `force_gpu=True`: Forces GPU mode (will attempt GPU operations even without GPU)

## Quick Start

### 1. Basic Testing

Run the quick test script:

```bash
python quick_test_hdbscan.py
```

This will test:
- CPU mode (forced)
- GPU mode (forced) 
- Auto mode (default)

### 2. Comprehensive Testing

Run the full test suite:

```bash
python test_hdbscan_scenarios.py
```

This includes:
- CPU with UMAP
- CPU without UMAP
- GPU with UMAP
- GPU without UMAP
- Auto mode
- Performance comparisons

### 3. Custom Testing

Use the clusterer directly in your code:

```python
import numpy as np
from app.nlp.hdbscan_clusterer import HDBSCANClusterer

# Generate test data
embeddings = np.random.randn(100, 32)

# Force CPU mode
cpu_clusterer = HDBSCANClusterer(
    force_cpu=True,
    use_umap=True,
    min_cluster_size=5
)

# Force GPU mode
gpu_clusterer = HDBSCANClusterer(
    force_gpu=True,
    use_umap=True,
    min_cluster_size=5
)

# Test clustering
cpu_labels = cpu_clusterer.cluster(embeddings)
gpu_labels = await gpu_clusterer.cluster_async(embeddings)
```

## Testing Scenarios

### CPU Mode Testing

```python
# Force CPU mode
clusterer = HDBSCANClusterer(force_cpu=True)

# This will:
# 1. Use CPU UMAP for dimensionality reduction
# 2. Use CPU HDBSCAN for clustering
# 3. Store the model for accessing properties
```

### GPU Mode Testing

```python
# Force GPU mode
clusterer = HDBSCANClusterer(force_gpu=True)

# This will:
# 1. Attempt to use GPU UMAP (cuML)
# 2. Attempt to use GPU HDBSCAN (cuML)
# 3. Convert data between CPU and GPU as needed
```

**Note**: GPU mode will fail if RAPIDS/cuML is not installed, but this allows you to test the GPU code path.

## Expected Behavior

### With Physical GPU + RAPIDS
- `force_cpu=True`: Uses CPU libraries
- `force_gpu=True`: Uses GPU libraries (cuML)
- Auto mode: Uses GPU if available

### Without Physical GPU or RAPIDS
- `force_cpu=True`: Uses CPU libraries ✅
- `force_gpu=True`: Fails with import errors ❌
- Auto mode: Uses CPU libraries ✅

## Testing Different Configurations

### UMAP vs No UMAP

```python
# With UMAP (dimensionality reduction)
clusterer_with_umap = HDBSCANClusterer(
    force_cpu=True,
    use_umap=True,
    umap_n_components=20
)

# Without UMAP (direct clustering)
clusterer_no_umap = HDBSCANClusterer(
    force_cpu=True,
    use_umap=False
)
```

### Different Clustering Parameters

```python
# Conservative clustering
conservative = HDBSCANClusterer(
    force_cpu=True,
    min_cluster_size=10,
    min_samples=5
)

# Aggressive clustering
aggressive = HDBSCANClusterer(
    force_cpu=True,
    min_cluster_size=3,
    min_samples=2
)
```

## Debugging

### Check Device Detection

```python
clusterer = HDBSCANClusterer(force_cpu=True)
print(f"GPU enabled: {clusterer._gpu_enabled}")
print(f"Device: {clusterer.device}")
```

### Check Available Libraries

```python
from app.nlp.hdbscan_clusterer import _HAS_CPU_UMAP, _HAS_GPU_STACK

print(f"CPU UMAP available: {_HAS_CPU_UMAP}")
print(f"GPU stack available: {_HAS_GPU_STACK}")
```

## Performance Testing

The test scripts include performance comparisons:

- Duration measurements
- Cluster count analysis
- Memory usage (for GPU mode)
- Speedup calculations

## Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies
   ```bash
   pip install umap-learn  # For CPU UMAP
   pip install cuml-cu11   # For GPU (if you have CUDA)
   ```

2. **GPU Mode Fails**: This is expected without RAPIDS/cuML
   - Use `force_cpu=True` for testing
   - Install RAPIDS for actual GPU testing

3. **Memory Issues**: Reduce test data size
   ```python
   embeddings = np.random.randn(50, 16)  # Smaller dataset
   ```

### Logging

Enable detailed logging to see what's happening:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Example Output

```
Generating test data...
Test data shape: (100, 32)

==================================================
Testing CPU Scenario (Forced)
==================================================
CPU clusterer created successfully
GPU enabled: False
CPU clustering completed!
Number of clusters: 3
Labels: [-1  0  1  2]

==================================================
Testing GPU Scenario (Forced)
==================================================
GPU clusterer created successfully
GPU enabled: True
GPU test failed: No module named 'cuml'
This is expected if you don't have RAPIDS/cuML installed
```

This testing approach allows you to verify both code paths work correctly, even without physical GPU hardware.
