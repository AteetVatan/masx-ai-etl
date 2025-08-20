# MASX AI ETL Concurrency Architecture

## Overview

This document describes the comprehensive concurrency refactor that eliminates multithreading anti-patterns and implements a unified, single-source concurrency runtime for optimal performance across CPU and GPU workloads.

## Architecture Principles

### 1. Single Source of Truth
- **Global GPU Detection**: All device selection logic centralized in `app/core/concurrency/device.py`
- **Unified Runtime**: Single `InferenceRuntime` class for all inference operations
- **Shared Executors**: One thread pool and one process pool per process

### 2. Async-First Design
- **Async I/O Everywhere**: HTTP, database, and file operations use async libraries
- **Event Loop Unblocked**: CPU-intensive work delegated to appropriate executors
- **No Nested Loops**: Proper async/await patterns throughout

### 3. Resource Management
- **Bounded Concurrency**: Semaphores and queue limits prevent resource exhaustion
- **Graceful Shutdown**: Proper cleanup of all resources and pending tasks
- **Backpressure**: Queue depth monitoring and automatic throttling

## Core Components

### Device Detection (`app/core/concurrency/device.py`)

```python
from app.core.concurrency.device import use_gpu, get_device_config

# Check if GPU should be used
if use_gpu():
    print("GPU path enabled")
else:
    print("CPU path enabled")

# Get complete device configuration
config = get_device_config()
print(f"Device: {config.device_type}:{config.device_id}")
```

**Environment Variables:**
- `MASX_FORCE_CPU=1`: Force CPU usage regardless of GPU availability
- `MASX_FORCE_GPU=1`: Force GPU usage (fails if CUDA unavailable)

### CPU Executors (`app/core/concurrency/cpu_executors.py`)

```python
from app.core.concurrency.cpu_executors import run_in_thread, run_in_process

# Run blocking function in thread pool
result = await run_in_thread(blocking_function, arg1, arg2)

# Run CPU-intensive function in process pool
result = await run_in_process(cpu_intensive_function, arg1, arg2)

# Batch processing
results = await map_threads(process_function, items)
results = await map_processes(process_function, items)
```

**Configuration:**
- `MAX_THREADS`: Maximum thread pool size (default: 20)
- `MAX_PROCS`: Maximum process pool size (default: 4)

### GPU Worker (`app/core/concurrency/gpu_worker.py`)

```python
from app.core.concurrency.gpu_worker import GPUWorker, GPUConfig

config = GPUConfig(
    max_batch_size=32,
    max_delay_ms=50,
    use_fp16=True
)

worker = GPUWorker(model_loader, config)
await worker.start()

# Single inference
result = await worker.infer(payload)

# Batch inference
results = await worker.infer_many(payloads)
```

**Configuration:**
- `BATCH_SIZE`: Maximum items per batch (default: 32)
- `BATCH_DELAY_MS`: Max delay before forcing batch (default: 50ms)
- `BATCH_QUEUE_MAX`: Maximum pending requests (default: 1000)

### Inference Runtime (`app/core/concurrency/runtime.py`)

```python
from app.core.concurrency.runtime import InferenceRuntime, RuntimeConfig

config = RuntimeConfig(
    gpu_batch_size=32,
    gpu_max_delay_ms=50,
    cpu_max_threads=20
)

runtime = InferenceRuntime(model_loader, config)
await runtime.start()

# Automatic device selection
result = await runtime.infer(payload)
results = await runtime.infer_many(payloads)

await runtime.stop()
```

## Usage Patterns

### 1. ETL Task Refactoring

**Before (Anti-pattern):**
```python
def process_feeds(self, feeds):
    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        futures = [executor.submit(self.process_feed, feed) for feed in feeds]
        results = [future.result() for future in futures]
    return results
```

**After (Correct pattern):**
```python
async def process_feeds(self, feeds):
    # Use inference runtime for model operations
    if self.inference_runtime:
        payloads = [{"feed": feed} for feed in feeds]
        results = await self.inference_runtime.infer_many(payloads)
        return results
    
    # Use CPU executors for other operations
    tasks = [
        self.cpu_executors.run_in_thread(self.process_feed, feed)
        for feed in feeds
    ]
    return await asyncio.gather(*tasks)
```

### 2. Web Scraping with Render Worker

```python
from app.infra_services.render import RenderClient

async with RenderClientContext() as client:
    # Single page
    response = await client.render_page("https://example.com")
    
    # Multiple pages
    responses = await client.render_pages([
        "https://example.com/1",
        "https://example.com/2"
    ])
    
    # With fallback
    response = await client.render_with_fallback(
        "https://example.com",
        primary_method="playwright",
        fallback_method="crawl4ai"
    )
```

### 3. Model Inference

```python
from app.core.concurrency import InferenceRuntime

# Create runtime with model loader
def load_model():
    return ModelManager.get_summarization_model()

runtime = InferenceRuntime(model_loader=load_model)
await runtime.start()

# Inference automatically routes to GPU or CPU
results = await runtime.infer_many(texts)

await runtime.stop()
```

## Configuration

### Environment Variables

```bash
# Device selection
export MASX_FORCE_CPU=1      # Force CPU usage
export MASX_FORCE_GPU=1      # Force GPU usage

# GPU settings
export BATCH_SIZE=32          # GPU batch size
export BATCH_DELAY_MS=50      # Max batch delay (ms)
export BATCH_QUEUE_MAX=1000   # Max queue size

# CPU settings
export MAX_THREADS=20         # Thread pool size
export MAX_PROCS=4            # Process pool size

# Render settings
export RENDER_MAX_PAGES=5     # Max concurrent pages
export RENDER_QUEUE_MAX=100   # Max render queue
export RENDER_TIMEOUT_S=30    # Render timeout
export RENDER_RETRIES=3       # Render retries
```

### Settings Integration

```python
from app.config import get_settings

settings = get_settings()

# GPU configuration
gpu_batch_size = settings.gpu_batch_size
gpu_max_delay_ms = settings.gpu_max_delay_ms
gpu_use_fp16 = settings.gpu_use_fp16

# CPU configuration
cpu_max_threads = settings.cpu_max_threads
cpu_max_processes = settings.cpu_max_processes

# Render configuration
render_max_pages = settings.render_max_pages
render_timeout_s = settings.render_timeout_s
```

## Performance Tuning

### GPU Optimization

1. **Batch Size Tuning**
   ```python
   # Larger batches = higher throughput, higher latency
   config = GPUConfig(max_batch_size=64, max_delay_ms=100)
   ```

2. **Memory Management**
   ```python
   # Enable FP16 for memory efficiency
   config = GPUConfig(use_fp16=True, pinned_memory=True)
   ```

3. **Queue Management**
   ```python
   # Monitor queue depth
   depth = runtime.queue_depth
   if depth > 100:
       print("High queue depth, consider increasing batch size")
   ```

### CPU Optimization

1. **Thread vs Process Selection**
   ```python
   # I/O bound work
   result = await run_in_thread(io_function, arg)
   
   # CPU bound work
   result = await run_in_process(cpu_function, arg)
   ```

2. **Concurrency Limits**
   ```python
   # Adjust based on system resources
   export MAX_THREADS=40      # For I/O heavy workloads
   export MAX_PROCS=8         # For CPU heavy workloads
   ```

## Monitoring and Metrics

### Runtime Metrics

```python
# Get comprehensive metrics
metrics = runtime.get_metrics()

print(f"Total requests: {metrics['total_requests']}")
print(f"GPU requests: {metrics['total_gpu_requests']}")
print(f"CPU requests: {metrics['total_cpu_requests']}")
print(f"Throughput: {metrics['requests_per_second']:.1f} req/s")
```

### GPU Worker Metrics

```python
gpu_metrics = worker.get_metrics()

print(f"Total inferences: {gpu_metrics['total_inferences']}")
print(f"Avg batch time: {gpu_metrics['avg_batch_time']:.3f}s")
print(f"Queue depth: {gpu_metrics['queue_size']}")
```

### Render Worker Metrics

```python
render_metrics = client.get_metrics()

print(f"Total renders: {render_metrics['worker_metrics']['total_renders']}")
print(f"Success rate: {render_metrics['worker_metrics']['success_rate']:.1%}")
print(f"Active pages: {render_metrics['worker_metrics']['active_pages']}")
```

## Error Handling

### Graceful Degradation

```python
try:
    # Try GPU path first
    result = await runtime.infer(payload)
except RuntimeError as e:
    if "GPU not available" in str(e):
        # Fallback to CPU
        result = await cpu_executors.run_in_thread(cpu_inference, payload)
    else:
        raise
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def robust_inference(payload):
    return await runtime.infer(payload)
```

## Migration Guide

### 1. Replace ThreadPoolExecutor

**Before:**
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(func, arg) for arg in args]
    results = [future.result() for future in futures]
```

**After:**
```python
from app.core.concurrency.cpu_executors import map_threads

results = await map_threads(func, args)
```

### 2. Replace ProcessPoolExecutor

**Before:**
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(func, arg) for arg in args]
    results = [future.result() for future in futures]
```

**After:**
```python
from app.core.concurrency.cpu_executors import map_processes

results = await map_processes(func, args)
```

### 3. Replace Direct GPU Calls

**Before:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

**After:**
```python
from app.core.concurrency.device import get_torch_device

device = get_torch_device()
model = model.to(device)
```

## Best Practices

### 1. Always Use Async

```python
# Good
async def process_data():
    result = await runtime.infer(data)
    return result

# Bad
def process_data():
    return asyncio.run(runtime.infer(data))
```

### 2. Proper Resource Cleanup

```python
# Good
async def main():
    runtime = InferenceRuntime()
    try:
        await runtime.start()
        result = await runtime.infer(data)
        return result
    finally:
        await runtime.stop()

# Bad
runtime = InferenceRuntime()
await runtime.start()
# ... use runtime ...
# Runtime never stopped!
```

### 3. Batch Processing

```python
# Good - Process in batches
batch_size = 32
for i in range(0, len(items), batch_size):
    batch = items[i:i + batch_size]
    results = await runtime.infer_many(batch)

# Bad - Process one by one
for item in items:
    result = await runtime.infer(item)
```

### 4. Error Boundaries

```python
async def safe_inference(payloads):
    try:
        return await runtime.infer_many(payloads)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        # Fallback to CPU
        return await cpu_executors.map_threads(cpu_inference, payloads)
```

## Troubleshooting

### Common Issues

1. **Event Loop Already Running**
   ```python
   # Solution: Use asyncio.run() only at top level
   asyncio.run(main_async_function())
   ```

2. **GPU Out of Memory**
   ```python
   # Solution: Reduce batch size
   config = GPUConfig(max_batch_size=16)
   ```

3. **High Latency**
   ```python
   # Solution: Increase batch size, reduce delay
   config = GPUConfig(max_batch_size=64, max_delay_ms=25)
   ```

4. **Thread Pool Exhaustion**
   ```python
   # Solution: Increase thread limit
   export MAX_THREADS=40
   ```

### Debug Mode

```python
import logging

# Enable debug logging
logging.getLogger("app.core.concurrency").setLevel(logging.DEBUG)

# Check runtime status
print(f"Runtime started: {runtime.is_started}")
print(f"Execution path: {runtime.execution_path}")
print(f"Queue depth: {runtime.queue_depth}")
```

## Performance Benchmarks

### Expected Improvements

- **Throughput**: 2-5x improvement with proper batching
- **Latency**: 20-50% reduction with async I/O
- **Resource Usage**: 30-60% reduction in memory and CPU
- **Scalability**: Linear scaling with proper concurrency limits

### Benchmarking

```python
import time
import asyncio

async def benchmark_runtime():
    runtime = InferenceRuntime(model_loader)
    await runtime.start()
    
    # Warmup
    await runtime.infer(dummy_data)
    
    # Benchmark
    start = time.time()
    results = await runtime.infer_many(test_data)
    duration = time.time() - start
    
    print(f"Processed {len(results)} items in {duration:.2f}s")
    print(f"Throughput: {len(results)/duration:.1f} items/s")
    
    await runtime.stop()
```

## Conclusion

This concurrency architecture provides:

1. **Elimination of anti-patterns**: No more nested thread pools
2. **Optimal performance**: GPU micro-batching + CPU parallelization
3. **Resource efficiency**: Shared executors and proper cleanup
4. **Developer experience**: Simple async API with automatic device selection
5. **Production readiness**: Monitoring, metrics, and error handling

The refactor maintains backward compatibility while providing a foundation for scalable, high-performance ETL operations.
