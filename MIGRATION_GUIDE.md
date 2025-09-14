# ModelManager Migration Guide

This guide provides step-by-step instructions for migrating from the singleton `ModelManager` to the new instance-based model management system.

## Overview

The new system replaces the singleton pattern with:
- **Instance-based managers**: Each manager is a regular class instance
- **Model pooling**: Controlled concurrency with limited model instances
- **Memory management**: Explicit cleanup and resource management
- **GPU optimization**: Dynamic pool sizing based on available VRAM

## Key Benefits

1. **Better Concurrency**: Multiple threads can use different model instances
2. **Memory Control**: Explicit cleanup prevents memory leaks
3. **GPU Optimization**: Automatic pool sizing based on available VRAM
4. **Resource Management**: Timeout support and graceful error handling
5. **Testability**: Easier to mock and test individual components

## Migration Steps

### Step 1: Update Imports

**Before:**
```python
from app.singleton import ModelManager
```

**After:**
```python
from app.core.models import SummarizationModelManager, EmbeddingModelManager
from app.config import get_settings
```

### Step 2: Create Manager Instances

**Before:**
```python
# Models were accessed directly from singleton
model, tokenizer, device = ModelManager.get_summarization_model()
```

**After:**
```python
# Create manager instance
settings = get_settings()
summarizer = SummarizationModelManager(settings)
embedder = EmbeddingModelManager(settings)
```

### Step 3: Use Context Managers (Recommended)

**Before:**
```python
model, tokenizer, device = ModelManager.get_summarization_model()
# ... use model directly ...
```

**After:**
```python
with summarizer.acquire() as instance:
    summary = summarizer.summarize_text(text, instance)
```

### Step 4: Handle Explicit Acquire/Release

**Before:**
```python
model, tokenizer, device = ModelManager.get_summarization_model()
# ... use model ...
# No cleanup needed (singleton persists)
```

**After:**
```python
instance = summarizer.get_instance(timeout=10.0)
try:
    summary = summarizer.summarize_text(text, instance)
finally:
    summarizer.release_instance(instance)
```

### Step 5: Update Class-based Usage

**Before:**
```python
class Summarizer:
    def __init__(self):
        self.model, self.tokenizer, self.device = ModelManager.get_summarization_model()
    
    def summarize(self, text):
        # ... use self.model, self.tokenizer, self.device ...
```

**After:**
```python
class Summarizer:
    def __init__(self):
        settings = get_settings()
        self.manager = SummarizationModelManager(settings)
    
    def summarize(self, text):
        with self.manager.acquire() as instance:
            return self.manager.summarize_text(text, instance)
```

## Configuration

### Settings Updates

Add these settings to your `settings.py`:

```python
# Model Pool Configuration
model_pool_enabled: bool = Field(default=True, description="Enable model pool")
model_pool_max_instances: int = Field(default=2, description="Maximum number of model instances")

# GPU Memory Estimation (optional)
model_vram_estimates: Dict[str, int] = Field(
    default={
        "summarization": 2 * 1024**3,  # 2GB for BART-large
        "embedding": 1 * 1024**3,      # 1GB for sentence-transformers
    },
    description="VRAM estimates per model type in bytes"
)
```

### Environment Variables

```bash
# Model pool settings
MODEL_POOL_MAX_INSTANCES=4
MODEL_POOL_ENABLED=true

# GPU settings
MASX_FORCE_GPU=false
MASX_FORCE_CPU=true
```

## Common Patterns

### 1. Basic Usage

```python
from app.core.models import SummarizationModelManager
from app.config import get_settings

settings = get_settings()
summarizer = SummarizationModelManager(settings)

with summarizer.acquire() as instance:
    summary = summarizer.summarize_text(text, instance)
```

### 2. Concurrent Processing

```python
import threading
from app.core.models import SummarizationModelManager
from app.config import get_settings

def worker(text):
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    with summarizer.acquire() as instance:
        return summarizer.summarize_text(text, instance)

# Start multiple threads
threads = []
for text in texts:
    thread = threading.Thread(target=worker, args=(text,))
    threads.append(thread)
    thread.start()
```

### 3. Memory Management

```python
# Automatic cleanup after use
with summarizer.acquire(destroy_after_use=True) as instance:
    summary = summarizer.summarize_text(text, instance)

# Manual cleanup
summarizer.cleanup()  # Destroy all instances
```

### 4. Error Handling

```python
try:
    with summarizer.acquire(timeout=5.0) as instance:
        summary = summarizer.summarize_text(text, instance)
except TimeoutError:
    print("No model instances available")
except Exception as e:
    print(f"Error: {e}")
```

## Testing

### Unit Tests

```python
from unittest.mock import Mock, patch
from app.core.models import SummarizationModelManager

def test_summarization():
    settings = Mock()
    settings.model_pool_max_instances = 2
    
    with patch('app.core.models.summarization_model_manager.AutoModelForSeq2SeqLM'):
        manager = SummarizationModelManager(settings)
        
        with manager.acquire() as instance:
            summary = manager.summarize_text("test text", instance)
            assert summary is not None
```

### Integration Tests

```python
def test_concurrent_usage():
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    def worker():
        with summarizer.acquire() as instance:
            return summarizer.summarize_text("test", instance)
    
    # Test concurrent access
    results = []
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
```

## Performance Considerations

### Pool Sizing

- **CPU**: Use `model_pool_max_instances` setting
- **GPU**: Automatically calculated based on available VRAM
- **Monitoring**: Use `manager.get_pool_stats()` to monitor usage

### Memory Optimization

```python
# For memory-intensive tasks
with summarizer.acquire(destroy_after_use=True) as instance:
    # Process large batch
    pass

# Shrink pool when done
summarizer.shrink_pool(1)
```

### GPU Optimization

```python
# Check GPU memory usage
stats = summarizer.get_pool_stats()
print(f"Pool size: {stats['max_instances']}")
print(f"Available instances: {stats['available']}")
```

## Troubleshooting

### Common Issues

1. **TimeoutError**: Increase timeout or pool size
2. **Memory issues**: Use `destroy_after_use=True`
3. **GPU detection**: Check CUDA installation and drivers
4. **Import errors**: Ensure new modules are in Python path

### Debug Information

```python
# Get pool statistics
stats = manager.get_pool_stats()
print(f"Pool stats: {stats}")

# Check device
device = manager.get_device()
print(f"Using device: {device}")
```

## Migration Checklist

- [ ] Update imports to use new managers
- [ ] Replace singleton calls with manager instances
- [ ] Use context managers for automatic cleanup
- [ ] Add error handling for timeouts
- [ ] Update class constructors to use managers
- [ ] Update tests to mock new managers
- [ ] Configure pool sizes in settings
- [ ] Test concurrent usage patterns
- [ ] Monitor memory usage and pool statistics
- [ ] Update documentation and examples

## Support

For questions or issues during migration:
1. Check the examples in `examples/` directory
2. Review unit tests in `tests/` directory
3. Monitor pool statistics and memory usage
4. Use mock models for testing without heavy downloads
