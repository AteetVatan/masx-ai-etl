# How to Use CPU/GPU Concurrency Step by Step

## Introduction

This guide walks you through using the new concurrency system in MASX AI ETL. It's designed for beginners and shows you how to replace the old multithreading patterns with modern, efficient async concurrency.

## Prerequisites

- Basic understanding of Python async/await syntax
- Familiarity with the existing ETL pipeline
- Access to the refactored codebase

## Step 1: Understanding the New Architecture

### What Changed?

**Before (Old Way):**
- Each module created its own `ThreadPoolExecutor`
- Mixed sync/async patterns with `asyncio.run()` inside threads
- Scattered GPU detection across multiple files
- No centralized resource management

**After (New Way):**
- Single `InferenceRuntime` for all inference operations
- Centralized device detection in `app/core/concurrency/device.py`
- Shared CPU executors (one thread pool + one process pool per process)
- Proper async/await patterns throughout

### Key Benefits

1. **Better Performance**: GPU micro-batching + CPU parallelization
2. **Resource Efficiency**: No more resource waste from multiple executors
3. **Easier Debugging**: Centralized logging and metrics
4. **Automatic Device Selection**: GPU when available, CPU fallback

## Step 2: Basic Setup

### Environment Variables

Set these in your `.env` file or environment:

```bash
# Force CPU usage (useful for development)
export MASX_FORCE_CPU=1

# Force GPU usage (will fail if CUDA unavailable)
export MASX_FORCE_GPU=1

# GPU settings
export BATCH_SIZE=32
export BATCH_DELAY_MS=50

# CPU settings
export MAX_THREADS=20
export MAX_PROCS=4
```

### Import the New Modules

```python
# For device detection
from app.core.concurrency.device import use_gpu, get_device_config

# For CPU operations
from app.core.concurrency.cpu_executors import run_in_thread, run_in_process

# For inference operations
from app.core.concurrency.runtime import InferenceRuntime, RuntimeConfig

# For web rendering
from app.infra_services.render import RenderClient
```

## Step 3: Simple Device Detection

### Check GPU Availability

```python
from app.core.concurrency.device import use_gpu

if use_gpu():
    print("🎉 GPU is available and will be used!")
else:
    print("💻 Using CPU for inference")
```

### Get Device Details

```python
from app.core.concurrency.device import get_device_config

config = get_device_config()
print(f"Device Type: {config.device_type}")
print(f"Device ID: {config.device_id}")
print(f"CUDA Available: {config.cuda_available}")
```

## Step 4: Using CPU Executors

### Replace ThreadPoolExecutor

**Old Way:**
```python
from concurrent.futures import ThreadPoolExecutor

def process_data(items):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_item, item) for item in items]
        results = [future.result() for future in futures]
    return results
```

**New Way:**
```python
from app.core.concurrency.cpu_executors import map_threads

async def process_data(items):
    results = await map_threads(process_item, items)
    return results
```

### Replace ProcessPoolExecutor

**Old Way:**
```python
from concurrent.futures import ProcessPoolExecutor

def heavy_computation(data):
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(compute, item) for item in data]
        results = [future.result() for future in futures]
    return results
```

**New Way:**
```python
from app.core.concurrency.cpu_executors import map_processes

async def heavy_computation(data):
    results = await map_processes(compute, data)
    return results
```

### Individual Function Execution

```python
from app.core.concurrency.cpu_executors import run_in_thread, run_in_process

# For I/O bound work (file reading, HTTP requests)
result = await run_in_thread(read_file, "data.txt")

# For CPU bound work (math, data processing)
result = await run_in_process(complex_calculation, data)
```

## Step 5: Using the Inference Runtime

### Basic Setup

```python
from app.core.concurrency.runtime import InferenceRuntime, RuntimeConfig

# Define how to load your model
def load_my_model():
    from app.singleton.model_manager import ModelManager
    return ModelManager.get_summarization_model()

# Create runtime configuration
config = RuntimeConfig(
    gpu_batch_size=32,        # How many items to process together
    gpu_max_delay_ms=50,      # Max wait time before processing batch
    cpu_max_threads=20        # Max CPU threads
)

# Create and start the runtime
runtime = InferenceRuntime(
    model_loader=load_my_model,
    config=config
)

await runtime.start()
```

### Single Inference

```python
# The runtime automatically chooses GPU or CPU
result = await runtime.infer("Your input data here")
print(f"Result: {result}")
```

### Batch Inference

```python
# Process multiple items efficiently
inputs = ["data1", "data2", "data3", "data4"]
results = await runtime.infer_many(inputs)

for input_data, result in zip(inputs, results):
    print(f"Input: {input_data} -> Result: {result}")
```

### Cleanup

```python
# Always stop the runtime when done
await runtime.stop()
```

## Step 6: Web Scraping with Render Worker

### Basic Usage

```python
from app.infra_services.render import RenderClient

# Create a render client
client = RenderClient()
await client.connect()

try:
    # Render a single page
    response = await client.render_page("https://example.com")
    
    if response.status == "success":
        print(f"Title: {response.content['title']}")
        print(f"Text: {response.content['text'][:100]}...")
    else:
        print(f"Error: {response.error}")
        
finally:
    # Always disconnect
    await client.disconnect()
```

### Using Context Manager

```python
from app.infra_services.render import RenderClientContext

async with RenderClientContext() as client:
    # Render multiple pages
    urls = [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3"
    ]
    
    responses = await client.render_pages(urls)
    
    for url, response in zip(urls, responses):
        if response.status == "success":
            print(f"✅ {url}: {len(response.content['text'])} characters")
        else:
            print(f"❌ {url}: {response.error}")
```

### With Fallback

```python
# Try Playwright first, fallback to Crawl4AI
response = await client.render_with_fallback(
    "https://example.com",
    primary_method="playwright",
    fallback_method="crawl4ai"
)
```

## Step 7: Refactoring Existing Code

### Step 7.1: Identify Old Patterns

Look for these in your code:

```python
# ❌ Old patterns to replace
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(func, item) for item in items]
    results = [future.result() for future in futures]

# ❌ Mixed async/sync
asyncio.run(async_function())  # Inside threaded functions
```

### Step 7.2: Replace with New Patterns

```python
# ✅ New patterns
from app.core.concurrency.cpu_executors import map_threads, map_processes

# For I/O bound work
results = await map_threads(func, items)

# For CPU bound work
results = await map_processes(func, items)
```

### Step 7.3: Update Function Signatures

**Before:**
```python
def process_data(self, items):
    # ... processing logic
    return results
```

**After:**
```python
async def process_data(self, items):
    # ... processing logic
    return results
```

### Step 7.4: Update Call Sites

**Before:**
```python
results = self.process_data(items)
```

**After:**
```python
results = await self.process_data(items)
```

## Step 8: Error Handling

### Basic Error Handling

```python
try:
    result = await runtime.infer(data)
except RuntimeError as e:
    if "GPU not available" in str(e):
        print("GPU failed, falling back to CPU")
        result = await run_in_thread(cpu_inference, data)
    else:
        raise
except Exception as e:
    print(f"Unexpected error: {e}")
    raise
```

### Graceful Degradation

```python
async def robust_inference(data):
    try:
        # Try GPU first
        return await runtime.infer(data)
    except Exception as e:
        print(f"GPU inference failed: {e}")
        
        try:
            # Fallback to CPU
            return await run_in_thread(cpu_inference, data)
        except Exception as e2:
            print(f"CPU inference also failed: {e2}")
            raise
```

## Step 9: Monitoring and Debugging

### Check Runtime Status

```python
# Basic status
print(f"Runtime started: {runtime.is_started}")
print(f"Execution path: {runtime.execution_path}")
print(f"Queue depth: {runtime.queue_depth}")

# Detailed metrics
metrics = runtime.get_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"Throughput: {metrics['requests_per_second']:.1f} req/s")
```

### Enable Debug Logging

```python
import logging

# Enable debug logging for concurrency modules
logging.getLogger("app.core.concurrency").setLevel(logging.DEBUG)
logging.getLogger("app.infra_services.render").setLevel(logging.DEBUG)
```

### Monitor Resource Usage

```python
# Check GPU worker status
if runtime.execution_path == "gpu":
    gpu_metrics = runtime.get_metrics()["gpu_worker"]
    print(f"GPU inferences: {gpu_metrics['total_inferences']}")
    print(f"Avg batch time: {gpu_metrics['avg_batch_time']:.3f}s")

# Check CPU executor status
cpu_metrics = runtime.get_metrics()["cpu_executors"]
print(f"Max threads: {cpu_metrics['max_threads']}")
print(f"Max processes: {cpu_metrics['max_processes']}")
```

## Step 10: Performance Tuning

### GPU Tuning

```python
# For higher throughput (higher latency)
config = RuntimeConfig(
    gpu_batch_size=64,        # Larger batches
    gpu_max_delay_ms=100      # Longer wait times
)

# For lower latency (lower throughput)
config = RuntimeConfig(
    gpu_batch_size=16,        # Smaller batches
    gpu_max_delay_ms=25       # Shorter wait times
)
```

### CPU Tuning

```python
# For I/O heavy workloads
export MAX_THREADS=40

# For CPU heavy workloads
export MAX_PROCS=8
```

### Monitor and Adjust

```python
# Check if queue is backing up
if runtime.queue_depth > 100:
    print("⚠️  High queue depth, consider increasing batch size")

# Check throughput
metrics = runtime.get_metrics()
if metrics['requests_per_second'] < 10:
    print("⚠️  Low throughput, check batch size and delays")
```

## Step 11: Testing Your Changes

### Unit Tests

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_inference_runtime():
    runtime = InferenceRuntime(model_loader)
    await runtime.start()
    
    try:
        result = await runtime.infer("test data")
        assert result is not None
    finally:
        await runtime.stop()

@pytest.mark.asyncio
async def test_cpu_executors():
    result = await run_in_thread(lambda x: x * 2, 5)
    assert result == 10
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_pipeline():
    # Test the entire ETL pipeline
    etl = ETLPipeline()
    results = await etl.run_all_etl_pipelines()
    
    assert len(results) > 0
    assert all(result is not None for result in results)
```

### Performance Tests

```python
import time

async def benchmark():
    runtime = InferenceRuntime(model_loader)
    await runtime.start()
    
    try:
        # Warmup
        await runtime.infer("warmup")
        
        # Benchmark
        start = time.time()
        results = await runtime.infer_many(["test"] * 100)
        duration = time.time() - start
        
        print(f"Processed 100 items in {duration:.2f}s")
        print(f"Throughput: {100/duration:.1f} items/s")
        
    finally:
        await runtime.stop()
```

## Step 12: Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Await

**❌ Wrong:**
```python
result = runtime.infer(data)  # Missing await!
```

**✅ Correct:**
```python
result = await runtime.infer(data)
```

### Pitfall 2: Not Starting the Runtime

**❌ Wrong:**
```python
runtime = InferenceRuntime(model_loader)
result = await runtime.infer(data)  # Runtime not started!
```

**✅ Correct:**
```python
runtime = InferenceRuntime(model_loader)
await runtime.start()
result = await runtime.infer(data)
```

### Pitfall 3: Not Cleaning Up

**❌ Wrong:**
```python
runtime = InferenceRuntime(model_loader)
await runtime.start()
result = await runtime.infer(data)
# Runtime never stopped!
```

**✅ Correct:**
```python
runtime = InferenceRuntime(model_loader)
try:
    await runtime.start()
    result = await runtime.infer(data)
    return result
finally:
    await runtime.stop()
```

### Pitfall 4: Using asyncio.run() Inside Async Functions

**❌ Wrong:**
```python
async def my_function():
    result = asyncio.run(other_async_function())  # Event loop already running!
    return result
```

**✅ Correct:**
```python
async def my_function():
    result = await other_async_function()
    return result
```

## Step 13: Production Deployment

### Environment Configuration

```bash
# Production settings
export MASX_FORCE_GPU=1          # Use GPU if available
export BATCH_SIZE=64              # Larger batches for production
export BATCH_DELAY_MS=100        # Longer delays for efficiency
export MAX_THREADS=40            # More threads for production
export MAX_PROCS=8               # More processes for production
```

### Health Checks

```python
async def health_check():
    try:
        runtime = InferenceRuntime(model_loader)
        await runtime.start()
        
        # Test inference
        result = await runtime.infer("health check")
        
        # Check metrics
        metrics = runtime.get_metrics()
        
        await runtime.stop()
        
        return {
            "status": "healthy",
            "execution_path": metrics["device_type"],
            "throughput": metrics["requests_per_second"]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

### Monitoring

```python
# Log metrics periodically
async def log_metrics():
    while True:
        try:
            metrics = runtime.get_metrics()
            logger.info(f"Runtime metrics: {metrics}")
            await asyncio.sleep(60)  # Log every minute
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
```

## Summary

You've now learned how to:

1. ✅ **Replace old multithreading patterns** with modern async concurrency
2. ✅ **Use the InferenceRuntime** for automatic GPU/CPU selection
3. ✅ **Leverage CPU executors** for I/O and CPU-bound work
4. ✅ **Implement web rendering** with isolated Playwright workers
5. ✅ **Handle errors gracefully** with fallback mechanisms
6. ✅ **Monitor performance** with comprehensive metrics
7. ✅ **Tune for production** with proper configuration

### Key Takeaways

- **Always use async/await** for better performance
- **Let the runtime choose** between GPU and CPU automatically
- **Process in batches** for optimal efficiency
- **Monitor queue depth** to prevent backpressure
- **Clean up resources** properly to avoid memory leaks
- **Test thoroughly** before deploying to production

### Next Steps

1. **Refactor your existing code** using the patterns shown
2. **Test with small datasets** before scaling up
3. **Monitor performance** and adjust configuration as needed
4. **Document your specific use cases** for team reference

The new concurrency system will give you better performance, easier debugging, and more reliable resource management. Happy coding! 🚀
