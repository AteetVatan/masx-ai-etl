# Multi-Model Instances Implementation for GPU Summarization

## Overview

This implementation allows the GPU worker to automatically calculate and load multiple model instances based on available GPU VRAM, then use them in parallel for summarization processing.

## Key Features

1. **Automatic VRAM Calculation**: Dynamically calculates how many model instances can fit in GPU memory
2. **Parallel Processing**: Distributes feeds across multiple model instances for concurrent processing
3. **Fallback Mechanisms**: Falls back to sequential processing if parallel processing fails
4. **Memory Optimization**: Reserves memory for batch data and activations

## Environment Variables (from settings.py)

### GPU Configuration
```bash
# Enable multiple model instances on single GPU
GPU_ENABLE_MULTI_MODEL_INSTANCES=true

# Maximum number of model instances per GPU (auto-calculated if None)
GPU_MAX_MODEL_INSTANCES=null

# GPU memory buffer for batch data and activations (GB)
GPU_MODEL_MEMORY_BUFFER_GB=2.0

# Estimated model size in GB (BART-large-CNN ~1.5GB FP16)
GPU_ESTIMATED_MODEL_SIZE_GB=1.5

# GPU batch size for inference
GPU_BATCH_SIZE=48

# GPU max delay before batch processing (ms)
GPU_MAX_DELAY_MS=50

# GPU inference queue size
GPU_QUEUE_SIZE=1000

# Use FP16 for GPU inference
GPU_USE_FP16=true

# Enable GPU model warmup
GPU_ENABLE_WARMUP=true
```

### Model Pool Configuration
```bash
# Enable model pool
MODEL_POOL_ENABLED=true

# Maximum number of model instances
MODEL_POOL_MAX_INSTANCES=1
```

## Implementation Details

### 1. VRAM Calculation

The system automatically calculates optimal model instances:

```python
def _calculate_optimal_model_instances(self) -> int:
    # Get GPU device
    device = torch.device(f"cuda:{self.config.device_id}")
    
    # Get total GPU memory
    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    available_memory_gb = torch.cuda.memory_allocated(device) / (1024**3)
    free_memory_gb = total_memory_gb - available_memory_gb
    
    # Reserve memory for batch data and activations
    reserved_memory_gb = self.config.model_memory_buffer_gb
    
    # Calculate usable memory for models
    usable_memory_gb = free_memory_gb - reserved_memory_gb
    
    # Calculate optimal instances
    optimal_instances = int(usable_memory_gb / estimated_model_size_gb)
    
    return max(1, optimal_instances)
```

### 2. Model Loading

Multiple model instances are loaded in parallel:

```python
async def _load_models(self):
    for i in range(self._optimal_model_instances):
        # Load model using the provided loader
        model = self.model_loader()
        
        # Move to GPU
        device = torch.device(f"cuda:{self.config.device_id}")
        model = model.to(device)
        
        # Enable FP16 if requested
        if self.config.use_fp16 and device.type == "cuda":
            model = model.half()
        
        self._models.append(model)
```

### 3. Parallel Processing

Feeds are distributed across model instances:

```python
def _process_summarization_output(self, batch_output: dict, batch_size: int):
    # Calculate chunk size for parallel processing
    num_models = len(self._models)
    feeds_per_model = max(1, len(original_payloads) // num_models)
    
    # Create tasks for parallel processing
    tasks = []
    for i in range(num_models):
        start_idx = i * feeds_per_model
        end_idx = start_idx + feeds_per_model if i < num_models - 1 else len(original_payloads)
        feeds_chunk = original_payloads[start_idx:end_idx]
        
        if feeds_chunk:
            model_instance = self._models[i % len(self._models)]
            task = process_feeds_with_model(model_instance, feeds_chunk)
            tasks.append(task)
    
    # Execute all tasks concurrently
    chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
```

## Example Configurations

### RTX 4090 (24GB VRAM)
```bash
# Auto-calculated: ~6-8 model instances
GPU_ESTIMATED_MODEL_SIZE_GB=1.5
GPU_MODEL_MEMORY_BUFFER_GB=2.0
GPU_MAX_MODEL_INSTANCES=null  # Auto-calculate
```

### RTX 3090 (24GB VRAM)
```bash
# Auto-calculated: ~6-8 model instances
GPU_ESTIMATED_MODEL_SIZE_GB=1.5
GPU_MODEL_MEMORY_BUFFER_GB=2.0
GPU_MAX_MODEL_INSTANCES=null  # Auto-calculate
```

### RTX 3080 (10GB VRAM)
```bash
# Auto-calculated: ~2-3 model instances
GPU_ESTIMATED_MODEL_SIZE_GB=1.5
GPU_MODEL_MEMORY_BUFFER_GB=2.0
GPU_MAX_MODEL_INSTANCES=null  # Auto-calculate
```

### RTX 3070 (8GB VRAM)
```bash
# Auto-calculated: ~1-2 model instances
GPU_ESTIMATED_MODEL_SIZE_GB=1.5
GPU_MODEL_MEMORY_BUFFER_GB=2.0
GPU_MAX_MODEL_INSTANCES=null  # Auto-calculate
```

## Performance Benefits

### 1. Parallel Processing
- **Multiple model instances** process different feeds simultaneously
- **Reduced total processing time** for large batches
- **Better GPU utilization** with concurrent operations

### 2. Memory Efficiency
- **Automatic calculation** prevents memory overflow
- **Reserved buffer** ensures stable performance
- **FP16 optimization** reduces memory usage

### 3. Scalability
- **Adapts to GPU capabilities** automatically
- **Handles varying batch sizes** efficiently
- **Fallback mechanisms** ensure reliability

## Usage Example

### 1. Environment Setup
```bash
# .env file
GPU_ENABLE_MULTI_MODEL_INSTANCES=true
GPU_MAX_MODEL_INSTANCES=null
GPU_MODEL_MEMORY_BUFFER_GB=2.0
GPU_ESTIMATED_MODEL_SIZE_GB=1.5
GPU_BATCH_SIZE=64
GPU_MAX_DELAY_MS=200
```

### 2. Code Usage
```python
from app.core.concurrency import GPUWorker, GPUConfig

# Create GPU worker with multiple model instances
config = GPUConfig(
    device_id=0,
    max_batch_size=64,
    enable_multi_model_instances=True
)

worker = GPUWorker(model_loader, config, TaskEnums.SUMMARIZER)
await worker.start()

# Process feeds using multiple model instances
results = await worker.infer_many(payloads)
```

## Monitoring and Metrics

### 1. GPU Memory Usage
```python
# Check GPU memory allocation
import torch
device = torch.device("cuda:0")
total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
free_memory = total_memory - allocated_memory

print(f"Total VRAM: {total_memory:.2f}GB")
print(f"Allocated: {allocated_memory:.2f}GB")
print(f"Free: {free_memory:.2f}GB")
```

### 2. Model Instance Count
```python
# Get number of loaded model instances
num_instances = len(worker._models)
print(f"Loaded model instances: {num_instances}")
```

### 3. Processing Performance
```python
# Get worker metrics
metrics = worker.get_metrics()
print(f"Total inferences: {metrics['total_inferences']}")
print(f"Throughput: {metrics['inferences_per_second']:.2f} inf/s")
```

## Best Practices

### 1. Memory Management
- Set appropriate `GPU_MODEL_MEMORY_BUFFER_GB` for your workload
- Monitor GPU memory usage during processing
- Use FP16 when possible to reduce memory footprint

### 2. Batch Sizing
- Larger batches work better with multiple model instances
- Balance between memory usage and processing efficiency
- Adjust `GPU_BATCH_SIZE` based on available VRAM

### 3. Error Handling
- Implement fallback mechanisms for failed model instances
- Monitor and log processing errors
- Use sequential fallback when parallel processing fails

## Troubleshooting

### 1. CUDA Out of Memory
```bash
# Reduce model instances
GPU_MAX_MODEL_INSTANCES=2

# Increase memory buffer
GPU_MODEL_MEMORY_BUFFER_GB=3.0

# Reduce batch size
GPU_BATCH_SIZE=32
```

### 2. Poor Performance
```bash
# Increase model instances
GPU_MAX_MODEL_INSTANCES=null  # Auto-calculate

# Optimize batch size
GPU_BATCH_SIZE=64

# Reduce delay
GPU_MAX_DELAY_MS=100
```

### 3. Model Loading Failures
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Check model loader function
# Ensure model_loader() returns valid model objects
```

## Conclusion

This implementation provides:
- **Automatic optimization** based on GPU capabilities
- **Parallel processing** for improved throughput
- **Memory efficiency** with automatic calculation
- **Reliability** with fallback mechanisms
- **Scalability** across different GPU configurations

The system automatically adapts to your hardware while providing optimal performance for summarization tasks.

