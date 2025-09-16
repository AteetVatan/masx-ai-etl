# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                       │
# │  Project: MASX AI – Strategic Agentic AI System               │
# │  All rights reserved.                                         │
# └───────────────────────────────────────────────────────────────┘
#
# MASX AI is a proprietary software system developed and owned by Ateet Vatan Bahmani.
# The source code, documentation, workflows, designs, and naming (including "MASX AI")
# are protected by applicable copyright and trademark laws.
#
# Redistribution, modification, commercial use, or publication of any portion of this
# project without explicit written consent is strictly prohibited.
#
# This project is not open-source and is intended solely for internal, research,
# or demonstration use by the author.
#
# Contact: ab@masxai.com | MASXAI.com

"""
Comprehensive unit tests for the new model management system.

Tests cover all functionality including pooling, concurrency, GPU detection,
memory management, and cleanup.
"""

import pytest
import threading
import time
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict

from app.core.model import (
    AbstractModel,
    ModelInstance,
    ModelPool_sync,
    SummarizationModelManager,
    EmbeddingModelManager
)


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, model_type: str = "test"):
        self.model_type = model_type
        self.cpu_called = False
        self.delete_called = False
    
    def cpu(self):
        self.cpu_called = True
    
    def delete(self):
        self.delete_called = True


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.pad_token = None
        self.eos_token = "eos"


class MockSettings:
    """Mock settings for testing."""
    
    def __init__(self, max_instances: int = 2):
        self.model_pool_max_instances = max_instances


class TestModelInstance:
    """Test ModelInstance functionality."""
    
    def test_model_instance_creation(self):
        """Test ModelInstance creation and basic properties."""
        model = MockModel()
        instance = ModelInstance(
            model=model,
            model_type="test",
            created_at=time.time()
        )
        
        assert instance.model == model
        assert instance.model_type == "test"
        assert not instance.in_use
        assert instance.created_at > 0
    
    def test_model_instance_destroy(self):
        """Test ModelInstance destroy method."""
        model = MockModel()
        instance = ModelInstance(model=model, model_type="test")
        
        instance.destroy()
        
        assert model.cpu_called
        assert model.delete_called


class TestModelPool:
    """Test ModelPool functionality."""
    
    def test_pool_creation(self):
        """Test ModelPool creation."""
        pool = ModelPool_sync(max_instances=3, model_type="test")
        
        assert pool.max_instances == 3
        assert pool.model_type == "test"
        assert pool.available.qsize() == 0
        assert len(pool.in_use) == 0
    
    def test_add_instance(self):
        """Test adding instances to pool."""
        pool = ModelPool_sync(max_instances=2, model_type="test")
        model = MockModel()
        instance = ModelInstance(model=model, model_type="test")
        
        result = pool.add_instance(instance)
        
        assert result is True
        assert pool.available.qsize() == 1
    
    def test_acquire_release(self):
        """Test acquiring and releasing instances."""
        pool = ModelPool_sync(max_instances=2, model_type="test")
        model = MockModel()
        instance = ModelInstance(model=model, model_type="test")
        pool.add_instance(instance)
        
        # Acquire instance
        acquired = pool.acquire(timeout=1.0)
        assert acquired == instance
        assert instance.in_use
        assert len(pool.in_use) == 1
        assert pool.available.qsize() == 0
        
        # Release instance
        pool.release(instance)
        assert not instance.in_use
        assert len(pool.in_use) == 0
        assert pool.available.qsize() == 1
    
    def test_acquire_timeout(self):
        """Test acquire timeout behavior."""
        pool = ModelPool_sync(max_instances=1, model_type="test")
        
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.1)
    
    def test_release_destroy(self):
        """Test releasing with destroy flag."""
        pool = ModelPool_sync(max_instances=2, model_type="test")
        model = MockModel()
        instance = ModelInstance(model=model, model_type="test")
        pool.add_instance(instance)
        
        acquired = pool.acquire()
        pool.release(acquired, destroy=True)
        
        assert model.cpu_called
        assert model.delete_called
        assert pool.available.qsize() == 0
    
    def test_get_stats(self):
        """Test pool statistics."""
        pool = ModelPool_sync(max_instances=3, model_type="test")
        
        # Add instances
        for i in range(2):
            model = MockModel()
            instance = ModelInstance(model=model, model_type="test")
            pool.add_instance(instance)
        
        # Acquire one
        acquired = pool.acquire()
        
        stats = pool.get_stats()
        assert stats["available"] == 1
        assert stats["in_use"] == 1
        assert stats["total"] == 2
        assert stats["max_instances"] == 3
    
    def test_shrink_pool(self):
        """Test pool shrinking."""
        pool = ModelPool_sync(max_instances=5, model_type="test")
        
        # Add multiple instances
        instances = []
        for i in range(4):
            model = MockModel()
            instance = ModelInstance(model=model, model_type="test")
            pool.add_instance(instance)
            instances.append(instance)
        
        # Shrink to 2 instances
        destroyed = pool.shrink_pool(2)
        
        assert destroyed == 2
        assert pool.available.qsize() == 2
        # Check that some models were destroyed
        destroyed_count = sum(1 for inst in instances if inst.model.cpu_called)
        assert destroyed_count == 2


class TestAbstractModelManager:
    """Test AbstractModelManager functionality."""
    
    class ConcreteModelManager(AbstractModel[MockModel]):
        """Concrete implementation for testing."""
        
        @property
        def model_name(self) -> str:
            return "test-model"
        
        @property
        def model_type(self) -> str:
            return "test"
        
        def _load_model(self) -> MockModel:
            return MockModel("test")
        
        def _load_tokenizer(self) -> Any:
            return MockTokenizer()
    
    def test_manager_creation(self):
        """Test manager creation."""
        settings = MockSettings()
        manager = self.ConcreteModelManager(settings)
        
        assert manager.settings == settings
        assert manager.model_name == "test-model"
        assert manager.model_type == "test"
        assert not manager._initialized
    
    def test_get_device_cpu(self):
        """Test device detection for CPU."""
        settings = MockSettings()
        manager = self.ConcreteModelManager(settings)
        
        with patch('app.core.concurrency.device.get_torch_device') as mock_get_device:
            mock_get_device.side_effect = Exception("No GPU")
            device = manager.get_device()
            
            assert device.type == "cpu"
    
    def test_get_device_gpu(self):
        """Test device detection for GPU."""
        settings = MockSettings()
        manager = self.ConcreteModelManager(settings)
        
        with patch('app.core.concurrency.device.get_torch_device') as mock_get_device:
            mock_get_device.return_value = torch.device("cuda:0")
            device = manager.get_device()
            
            assert device.type == "cuda"
    
    def test_calculate_pool_size_cpu(self):
        """Test pool size calculation for CPU."""
        settings = MockSettings(max_instances=4)
        manager = self.ConcreteModelManager(settings)
        
        with patch.object(manager, 'get_device', return_value=torch.device("cpu")):
            size = manager._calculate_pool_size()
            assert size == 4
    
    def test_calculate_pool_size_gpu(self):
        """Test pool size calculation for GPU."""
        settings = MockSettings(max_instances=4)
        manager = self.ConcreteModelManager(settings)
        
        with patch.object(manager, 'get_device', return_value=torch.device("cuda")):
            with patch.object(manager, '_get_gpu_memory_info') as mock_memory:
                mock_memory.return_value = {
                    'free_bytes': 8 * 1024**3,  # 8GB free
                    'total_bytes': 16 * 1024**3  # 16GB total
                }
                size = manager._calculate_pool_size()
                # Should calculate based on VRAM (8GB / 2GB per model = 4 instances)
                assert size == 4
    
    def test_gpu_memory_info_pynvml(self):
        """Test GPU memory info via pynvml."""
        settings = MockSettings()
        manager = self.ConcreteModelManager(settings)
        
        with patch('pynvml.nvmlInit'), \
             patch('pynvml.nvmlDeviceGetHandleByIndex') as mock_handle, \
             patch('pynvml.nvmlDeviceGetMemoryInfo') as mock_memory:
            
            mock_memory.return_value = Mock(
                total=16 * 1024**3,
                free=8 * 1024**3,
                used=8 * 1024**3
            )
            
            info = manager._get_gpu_memory_info()
            
            assert info is not None
            assert info['total_bytes'] == 16 * 1024**3
            assert info['free_bytes'] == 8 * 1024**3
    
    def test_gpu_memory_info_nvidia_smi(self):
        """Test GPU memory info via nvidia-smi."""
        settings = MockSettings()
        manager = self.ConcreteModelManager(settings)
        
        with patch('pynvml.nvmlInit', side_effect=ImportError), \
             patch('subprocess.run') as mock_run:
            
            mock_run.return_value = Mock(
                returncode=0,
                stdout="16384, 8192, 8192"  # total, free, used in MB
            )
            
            info = manager._get_gpu_memory_info()
            
            assert info is not None
            assert info['total_bytes'] == 16 * 1024**3
            assert info['free_bytes'] == 8 * 1024**3
    
    def test_initialize(self):
        """Test manager initialization."""
        settings = MockSettings(max_instances=3)
        manager = self.ConcreteModelManager(settings)
        
        with patch.object(manager, '_calculate_pool_size', return_value=2):
            manager.initialize()
            
            assert manager._initialized
            assert manager._pool is not None
            assert manager._pool.max_instances == 2
    
    def test_acquire_context_manager(self):
        """Test acquire context manager."""
        settings = MockSettings()
        manager = self.ConcreteModelManager(settings)
        
        with patch.object(manager, 'initialize'):
            manager._pool = ModelPool_sync(max_instances=1, model_type="test")
            model = MockModel()
            instance = ModelInstance(model=model, model_type="test")
            manager._pool.add_instance(instance)
            
            with manager.acquire() as acquired:
                assert acquired == instance
                assert instance.in_use
            
            assert not instance.in_use
    
    def test_acquire_context_manager_destroy(self):
        """Test acquire context manager with destroy."""
        settings = MockSettings()
        manager = self.ConcreteModelManager(settings)
        
        with patch.object(manager, 'initialize'):
            manager._pool = ModelPool_sync(max_instances=1, model_type="test")
            model = MockModel()
            instance = ModelInstance(model=model, model_type="test")
            manager._pool.add_instance(instance)
            
            with manager.acquire(destroy_after_use=True) as acquired:
                assert acquired == instance
            
            assert model.cpu_called
            assert model.delete_called
    
    def test_get_instance_explicit(self):
        """Test explicit acquire/release pattern."""
        settings = MockSettings()
        manager = self.ConcreteModelManager(settings)
        
        with patch.object(manager, 'initialize'):
            manager._pool = ModelPool_sync(max_instances=1, model_type="test")
            model = MockModel()
            instance = ModelInstance(model=model, model_type="test")
            manager._pool.add_instance(instance)
            
            acquired = manager.get_instance()
            assert acquired == instance
            assert instance.in_use
            
            manager.release_instance(acquired)
            assert not instance.in_use
    
    def test_get_pool_stats(self):
        """Test getting pool statistics."""
        settings = MockSettings()
        manager = self.ConcreteModelManager(settings)
        
        with patch.object(manager, 'initialize'):
            manager._pool = ModelPool_sync(max_instances=2, model_type="test")
            model = MockModel()
            instance = ModelInstance(model=model, model_type="test")
            manager._pool.add_instance(instance)
            
            stats = manager.get_pool_stats()
            assert stats["available"] == 1
            assert stats["in_use"] == 0
            assert stats["total"] == 1
    
    def test_cleanup(self):
        """Test manager cleanup."""
        settings = MockSettings()
        manager = self.ConcreteModelManager(settings)
        
        with patch.object(manager, 'initialize'):
            manager._pool = ModelPool_sync(max_instances=2, model_type="test")
            model = MockModel()
            instance = ModelInstance(model=model, model_type="test")
            manager._pool.add_instance(instance)
            
            manager.cleanup()
            
            assert model.cpu_called
            assert model.delete_called


class TestSummarizationModelManager:
    """Test SummarizationModelManager functionality."""
    
    def test_manager_creation(self):
        """Test SummarizationModelManager creation."""
        settings = MockSettings()
        manager = SummarizationModelManager(settings)
        
        assert manager.model_name == "facebook/bart-large-cnn"
        assert manager.model_type == "summarization"
        assert manager._get_model_vram_estimate() == 2 * 1024**3
    
    @patch('app.core.model.summarization_model_manager.AutoModelForSeq2SeqLM')
    @patch('app.core.model.summarization_model_manager.AutoTokenizer')
    def test_load_model(self, mock_tokenizer, mock_model):
        """Test model loading."""
        settings = MockSettings()
        manager = SummarizationModelManager(settings)
        
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        with patch.object(manager, 'get_device', return_value=torch.device("cpu")):
            with patch.object(manager, '_ensure_nltk_models'):
                model = manager._load_model()
                
                assert model == mock_model_instance
                mock_model.from_pretrained.assert_called_once()
    
    @patch('app.core.model.summarization_model_manager.AutoTokenizer')
    def test_load_tokenizer(self, mock_tokenizer):
        """Test tokenizer loading."""
        settings = MockSettings()
        manager = SummarizationModelManager(settings)
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "eos"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        tokenizer = manager._load_tokenizer()
        
        assert tokenizer == mock_tokenizer_instance
        assert tokenizer.pad_token == "eos"


class TestEmbeddingModelManager:
    """Test EmbeddingModelManager functionality."""
    
    def test_manager_creation(self):
        """Test EmbeddingModelManager creation."""
        settings = MockSettings()
        manager = EmbeddingModelManager(settings)
        
        assert manager.model_name == "sentence-transformers/all-mpnet-base-v2"
        assert manager.model_type == "embedding"
        assert manager._get_model_vram_estimate() == 1 * 1024**3
    
    @patch('app.core.model.embedding_model_manager.SentenceTransformer')
    def test_load_model(self, mock_transformer):
        """Test model loading."""
        settings = MockSettings()
        manager = EmbeddingModelManager(settings)
        
        mock_model_instance = Mock()
        mock_transformer.return_value = mock_model_instance
        
        with patch('torch.cuda.is_available', return_value=False):
            model = manager._load_model()
            
            assert model == mock_model_instance
            mock_transformer.assert_called_once()
    
    def test_load_tokenizer(self):
        """Test tokenizer loading (should return None)."""
        settings = MockSettings()
        manager = EmbeddingModelManager(settings)
        
        tokenizer = manager._load_tokenizer()
        assert tokenizer is None


class TestConcurrency:
    """Test concurrent access patterns."""
    
    def test_concurrent_acquire_release(self):
        """Test concurrent acquire/release operations."""
        pool = ModelPool_sync(max_instances=2, model_type="test")
        
        # Add instances
        for i in range(2):
            model = MockModel()
            instance = ModelInstance(model=model, model_type="test")
            pool.add_instance(instance)
        
        results = []
        errors = []
        
        def worker():
            try:
                instance = pool.acquire(timeout=2.0)
                time.sleep(0.1)  # Simulate work
                pool.release(instance)
                results.append("success")
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert len(results) == 5
    
    def test_concurrent_acquire_timeout(self):
        """Test concurrent acquire with timeout."""
        pool = ModelPool_sync(max_instances=1, model_type="test")
        
        # Add one instance
        model = MockModel()
        instance = ModelInstance(model=model, model_type="test")
        pool.add_instance(instance)
        
        results = []
        errors = []
        
        def worker():
            try:
                instance = pool.acquire(timeout=0.5)
                time.sleep(1.0)  # Hold for longer than timeout
                pool.release(instance)
                results.append("success")
            except TimeoutError:
                errors.append("timeout")
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have 1 success and 2 timeouts
        assert len([e for e in errors if e == "timeout"]) == 2
        assert len(results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
