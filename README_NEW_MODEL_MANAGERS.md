# New Model Management System

This document provides a comprehensive overview of the new model management system that replaces the singleton `ModelManager` with instance-based managers supporting controlled concurrency, memory management, and GPU optimization.

## 🎯 Purpose

The new system addresses key limitations of the singleton pattern:
- **Concurrency Issues**: Multiple threads competing for the same model instance
- **Memory Management**: No control over model lifecycle and cleanup
- **Resource Optimization**: No dynamic sizing based on available hardware
- **Testability**: Difficult to mock and test singleton components

## 🏗️ Design

### Architecture Overview

```
AbstractModelManager (Base Class)
├── SummarizationModelManager (BART models)
├── EmbeddingModelManager (sentence-transformers)
└── ModelPool (Thread-safe instance management)
```

### Key Components

1. **AbstractModelManager**: Base class with pooling and lifecycle management
2. **ModelPool**: Thread-safe pool with acquire/release semantics
3. **ModelInstance**: Container for model instances with metadata
4. **Concrete Managers**: Specialized managers for different model types

## 🚀 Features

### 1. Instance-Based Management
- Each manager is a regular class instance (not singleton)
- Multiple managers can coexist for different model types
- Easy to create, configure, and destroy

### 2. Model Pooling
- Controlled concurrency with limited model instances
- Thread-safe acquire/release operations
- Timeout support for graceful error handling
- Automatic pool sizing based on available resources

### 3. Memory Management
- Explicit cleanup and resource management
- Destroy instances after use to free memory
- Pool shrinking capabilities
- GPU memory optimization

### 4. GPU Optimization
- Dynamic pool sizing based on available VRAM
- Automatic GPU/CPU detection
- Fallback to CPU when GPU unavailable
- Support for multiple GPU configurations

### 5. Thread Safety
- Thread-safe pool operations
- Concurrent access to different model instances
- Proper locking and synchronization
- Race condition prevention

## 📁 File Structure

```
app/core/models/
├── __init__.py                    # Package exports
├── abstract_model_manager.py      # Base class and ModelPool
├── summarization_model_manager.py # BART summarization models
└── embedding_model_manager.py     # sentence-transformers models

tests/
├── __init__.py
├── test_model_managers.py         # Comprehensive unit tests
└── test_mock_models.py           # Mock implementations

examples/
├── __init__.py
├── model_manager_usage.py         # Usage examples
└── migration_guide.py            # Migration examples

MIGRATION_GUIDE.md                 # Detailed migration instructions
README_NEW_MODEL_MANAGERS.md      # This document
```

## 🔧 Usage Examples

### Basic Usage (Context Manager)

```python
from app.core.models import SummarizationModelManager
from app.config import get_settings

settings = get_settings()
summarizer = SummarizationModelManager(settings)

with summarizer.acquire() as instance:
    summary = summarizer.summarize_text(text, instance)
```

### Explicit Acquire/Release

```python
instance = summarizer.get_instance(timeout=10.0)
try:
    summary = summarizer.summarize_text(text, instance)
finally:
    summarizer.release_instance(instance)
```

### Concurrent Processing

```python
import threading

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

### Memory Management

```python
# Automatic cleanup
with summarizer.acquire(destroy_after_use=True) as instance:
    summary = summarizer.summarize_text(text, instance)

# Manual cleanup
summarizer.cleanup()  # Destroy all instances
```

## ⚙️ Configuration

### Settings

```python
# Model Pool Configuration
model_pool_enabled: bool = True
model_pool_max_instances: int = 2

# GPU Settings
masx_force_gpu: bool = False
masx_force_cpu: bool = True
gpu_use_fp16: bool = True
```

### Environment Variables

```bash
MODEL_POOL_MAX_INSTANCES=4
MASX_FORCE_GPU=false
MASX_FORCE_CPU=true
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run specific test
python run_tests.py test_concurrent_usage

# Run with pytest directly
pytest tests/test_model_managers.py -v
```

### Test Coverage

- ✅ Model instance creation and destruction
- ✅ Pool acquire/release operations
- ✅ Concurrent access patterns
- ✅ Timeout handling
- ✅ Memory management
- ✅ GPU detection and pool sizing
- ✅ Error handling and edge cases

## 📊 Performance

### Pool Statistics

```python
stats = manager.get_pool_stats()
print(f"Available instances: {stats['available']}")
print(f"Instances in use: {stats['in_use']}")
print(f"Total instances: {stats['total']}")
print(f"Max instances: {stats['max_instances']}")
```

### Memory Optimization

- **CPU**: Uses `model_pool_max_instances` setting
- **GPU**: Calculates based on available VRAM
- **Monitoring**: Real-time pool statistics
- **Cleanup**: Explicit destroy methods

## 🔄 Migration

### Step-by-Step Migration

1. **Update Imports**
   ```python
   # Before
   from app.singleton import ModelManager
   
   # After
   from app.core.models import SummarizationModelManager, EmbeddingModelManager
   ```

2. **Create Manager Instances**
   ```python
   settings = get_settings()
   summarizer = SummarizationModelManager(settings)
   ```

3. **Use Context Managers**
   ```python
   with summarizer.acquire() as instance:
       summary = summarizer.summarize_text(text, instance)
   ```

4. **Update Class Constructors**
   ```python
   class Summarizer:
       def __init__(self):
           settings = get_settings()
           self.manager = SummarizationModelManager(settings)
   ```

### Migration Checklist

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

## 🛠️ Development

### Adding New Model Types

1. **Create Manager Class**
   ```python
   class NewModelManager(AbstractModelManager[NewModel]):
       @property
       def model_name(self) -> str:
           return "new-model-name"
       
       @property
       def model_type(self) -> str:
           return "new-type"
       
       def _load_model(self) -> NewModel:
           # Load model implementation
           pass
   ```

2. **Implement Model Loading**
   ```python
   def _load_model(self) -> NewModel:
       # Model loading logic
       return model
   
   def _load_tokenizer(self) -> Optional[Any]:
       # Tokenizer loading logic
       return tokenizer
   ```

3. **Add to Package Exports**
   ```python
   # app/core/models/__init__.py
   from .new_model_manager import NewModelManager
   
   __all__ = [..., "NewModelManager"]
   ```

### Testing New Managers

```python
def test_new_manager():
    settings = MockSettings()
    manager = NewModelManager(settings)
    
    with manager.acquire() as instance:
        result = manager.process_data(data, instance)
        assert result is not None
```

## 🔍 Monitoring

### Pool Statistics

```python
# Get current pool status
stats = manager.get_pool_stats()
print(f"Pool utilization: {stats['in_use']}/{stats['max_instances']}")

# Monitor over time
import time
while True:
    stats = manager.get_pool_stats()
    print(f"Pool stats: {stats}")
    time.sleep(5)
```

### Memory Usage

```python
# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.1f}GB")
```

## 🚨 Troubleshooting

### Common Issues

1. **TimeoutError**: Increase timeout or pool size
2. **Memory issues**: Use `destroy_after_use=True`
3. **GPU detection**: Check CUDA installation
4. **Import errors**: Ensure modules are in Python path

### Debug Information

```python
# Enable debug logging
import logging
logging.getLogger("app.core.models").setLevel(logging.DEBUG)

# Check device detection
device = manager.get_device()
print(f"Using device: {device}")

# Monitor pool operations
stats = manager.get_pool_stats()
print(f"Pool stats: {stats}")
```

## 📚 Additional Resources

- **Migration Guide**: `MIGRATION_GUIDE.md`
- **Usage Examples**: `examples/model_manager_usage.py`
- **Migration Examples**: `examples/migration_guide.py`
- **Unit Tests**: `tests/test_model_managers.py`
- **Mock Models**: `tests/test_mock_models.py`
- **Demo Script**: `demo_new_model_managers.py`

## 🤝 Contributing

When contributing to the model management system:

1. **Follow the abstract base class pattern**
2. **Add comprehensive tests for new functionality**
3. **Update documentation and examples**
4. **Ensure thread safety for all operations**
5. **Add proper error handling and logging**
6. **Test with both CPU and GPU configurations**

## 📄 License

This code is part of the MASX AI system and is proprietary software developed by Ateet Vatan Bahmani. All rights reserved.
