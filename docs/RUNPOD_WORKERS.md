# RunPod Serverless Worker System for Parallel Execution

## Overview

The RunPod Serverless Worker System enables true parallel execution of ETL pipelines by distributing flashpoints across multiple RunPod Serverless instances. This significantly improves performance when processing large numbers of flashpoints by leveraging RunPod's built-in scaling capabilities.

## Configuration

### Environment Variables

Set the number of workers and RunPod configuration in your `.env` file:

```bash
# RunPod Worker Configuration
RUNPOD_WORKERS=4
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_ENDPOINT=https://api.runpod.io/v2/your_endpoint_id/run
```

### Settings

The worker count is automatically loaded from the `RUNPOD_WORKERS` environment variable and can be accessed via:

```python
from app.config import get_settings
settings = get_settings()
num_workers = settings.runpod_workers
```

## How It Works

### 1. Flashpoint Distribution

The `RunPodServerlessManager` automatically distributes flashpoints evenly across available RunPod Serverless instances:

- **Single Worker**: All flashpoints processed sequentially (original behavior)
- **Multiple Workers**: Flashpoints split into chunks and sent to different RunPod instances for true parallel processing

### 2. Parallel Execution

Each RunPod Serverless instance processes its assigned flashpoints independently:

```python
# Instance 1: flashpoints[0:3]  (Coordinator)
# Instance 2: flashpoints[3:6]  (Worker)
# Instance 3: flashpoints[6:9]  (Worker)
# Instance 4: flashpoints[9:12] (Worker)
```

### 3. Result Aggregation

Results from all RunPod Serverless instances are collected via API calls and aggregated into a single list, maintaining the original order.

## Usage

### Automatic Integration

The RunPod Serverless worker system is automatically integrated into the ETL pipeline. No code changes required:

```python
# In your ETL pipeline
worker_manager = RunPodServerlessManager(self.settings.runpod_workers)
results = await worker_manager.distribute_to_workers(
    flashpoints, 
    date=self.date,
    cleanup=True
)
```

### Manual Worker Management

You can also use the RunPod Serverless manager directly:

```python
from app.core.concurrency import RunPodServerlessManager

# Create worker manager
worker_manager = RunPodServerlessManager(num_workers=4)

# Distribute flashpoints
worker_chunks = worker_manager._distribute_flashpoints(flashpoints)

# Process with RunPod workers
results = await worker_manager.distribute_to_workers(
    flashpoints, 
    date="2025-01-01",
    cleanup=True
)
```

## Performance Benefits

### Speedup Calculation

- **Sequential Processing**: `total_time = flashpoints × processing_time_per_flashpoint`
- **Parallel Processing**: `total_time ≈ max(worker_chunk_times)`
- **Speedup**: `sequential_time / parallel_time`

### Example

With 4 workers processing 12 flashpoints:
- **Sequential**: 12 × 10s = 120 seconds
- **Parallel**: max(3 × 10s, 3 × 10s, 3 × 10s, 3 × 10s) = 30 seconds
- **Speedup**: 120s / 30s = 4x

## Best Practices

### 1. Worker Count Selection

- **CPU-bound tasks**: Set workers = number of CPU cores
- **I/O-bound tasks**: Set workers = 2-4x number of CPU cores
- **Memory constraints**: Reduce workers if memory usage is high

### 2. Resource Management

- Each worker maintains its own database connections
- Monitor memory usage per worker
- Consider worker pool limits for database connections

### 3. Error Handling

- Worker failures are automatically logged
- Failed flashpoints return `None` in results
- Overall pipeline continues even if individual workers fail

## Monitoring

### Logs

Worker activity is logged with worker IDs:

```
INFO: Worker 1 starting with 3 flashpoints
INFO: Worker 1 completed flashpoint test_flashpoint_0
INFO: Worker 1 completed flashpoint test_flashpoint_1
INFO: Worker 1 completed flashpoint test_flashpoint_2
INFO: Worker 1 completed all 3 flashpoints
```

### Metrics

The handler response includes worker count:

```json
{
  "ok": true,
  "date": "2025-01-01",
  "trigger": "manual",
  "cleanup": true,
  "workers": 4,
  "duration_sec": 45.2
}
```

## Testing

Run the test script to verify worker distribution:

```bash
python test_workers.py
```

This will demonstrate:
- Flashpoint distribution across workers
- Parallel processing simulation
- Performance metrics and speedup calculations

## Troubleshooting

### Common Issues

1. **No speedup observed**: Check if tasks are truly parallelizable
2. **Memory errors**: Reduce worker count
3. **Database connection errors**: Check connection pool limits

### Debug Mode

Enable debug logging to see detailed worker activity:

```bash
LOG_LEVEL=DEBUG
```

## Timeout Configuration for Long-Running ETL Processes

The system has been configured with extended timeouts to handle long-running ETL processes:

### **Updated Timeout Values**
- **Request Timeout**: 2 hours (7200 seconds) - For overall ETL pipeline execution
- **GPU Timeout**: 2 hours (7200 seconds) - For GPU inference operations
- **Render Timeout**: 1 hour (3600 seconds) - For complex web page rendering
- **Web Scraping Timeout**: 1 hour (3600 seconds) - For content extraction
- **Navigation Timeout**: 30 minutes (1800 seconds) - For page navigation

### **Why Extended Timeouts?**
- **ETL processes can take hours** to complete large datasets
- **Web scraping complex pages** may require extended processing time
- **GPU operations** on large batches can be time-consuming
- **Network delays** and retries in distributed environments

### **Configuration**
Timeouts are automatically configured via environment variables:
```bash
REQUEST_TIMEOUT=7200          # 2 hours
GPU_TIMEOUT=7200             # 2 hours  
RENDER_TIMEOUT_S=3600        # 1 hour
```

## How RunPod Serverless Integration Works

### 1. **Coordinator Instance**
- Receives the main ETL request
- Splits flashpoints into chunks based on `RUNPOD_WORKERS`
- Creates new RunPod Serverless instances via API calls
- Aggregates results from all workers

### 2. **Worker Instances**
- Each new instance receives a subset of flashpoints
- Processes flashpoints independently using the same ETL pipeline
- Returns results to the coordinator via API response
- Automatically terminated after completion

### 3. **Automatic Scaling**
- RunPod Serverless creates workers as needed
- You control max concurrent workers via `RUNPOD_WORKERS`
- Pay only for actual processing time
- Built-in load balancing and fault tolerance

## Migration from Single Worker

The system is backward compatible:

- **RUNPOD_WORKERS=1**: Original sequential behavior (local processing)
- **RUNPOD_WORKERS>1**: New parallel behavior (multiple RunPod instances)
- **No code changes required**: Automatic detection and switching
