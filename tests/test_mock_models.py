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
Mock model implementations for testing.

Provides lightweight mock implementations of HuggingFace models to avoid
heavy downloads during testing while maintaining realistic behavior.
"""

import torch
from typing import List, Union, Dict, Any
from unittest.mock import Mock
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from sentence_transformers import SentenceTransformer


class MockModel:
    """Base mock model class."""

    def __init__(self, model_type: str = "test"):
        self.model_type = model_type
        self.cpu_called = False
        self.delete_called = False
        self.eval_called = False
        self.config = Mock()
        self.config.use_cache = True

    def cpu(self):
        self.cpu_called = True
        return self

    def delete(self):
        self.delete_called = True

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        return self


class MockSummarizationModel(MockModel):
    """Mock BART summarization model."""

    def __init__(self):
        super().__init__("summarization")
        self.config = Mock()
        self.config.use_cache = True
        self.config.max_position_embeddings = 1024

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Mock generate method that returns fake summaries."""
        batch_size = input_ids.shape[0]
        # Return fake token IDs (shorter than input)
        fake_output = torch.randint(1, 1000, (batch_size, 50))
        return fake_output

    def get_input_embeddings(self):
        """Mock input embeddings."""
        embeddings = Mock()
        embeddings.num_embeddings = 50265  # BART vocab size
        return embeddings


class MockEmbeddingModel(MockModel):
    """Mock sentence-transformers model."""

    def __init__(self):
        super().__init__("embedding")
        self.embedding_dim = 768  # all-mpnet-base-v2 dimension

    def encode(
        self, texts, batch_size=32, convert_to_tensor=True, show_progress_bar=False
    ):
        """Mock encode method that returns fake embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        # Generate fake embeddings
        embeddings = torch.randn(len(texts), self.embedding_dim)

        if convert_to_tensor:
            return embeddings
        else:
            return embeddings.tolist()


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 50265):
        self.vocab_size = vocab_size
        self.pad_token = None
        self.eos_token = "eos"
        self.pad_token_id = 1
        self.eos_token_id = 2

    def __len__(self):
        return self.vocab_size

    def from_pretrained(self, model_name, **kwargs):
        """Mock from_pretrained method."""
        return self

    def encode(self, text, **kwargs):
        """Mock encode method."""
        # Return fake token IDs
        return torch.randint(1, 1000, (1, 20))

    def __call__(self, text, **kwargs):
        """Mock call method for tokenization."""
        if isinstance(text, str):
            text = [text]

        # Return fake tokenized input
        return {
            "input_ids": torch.randint(1, 1000, (len(text), 20)),
            "attention_mask": torch.ones(len(text), 20),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        """Mock decode method."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Return fake decoded text
        return "This is a mock summary of the input text."


class MockSummarizationModelManager:
    """Mock summarization model manager for testing."""

    def __init__(self, settings=None):
        self.settings = settings or Mock()
        self.settings.model_pool_max_instances = 2
        self._pool = None
        self._initialized = False

    def initialize(self):
        """Initialize the mock manager."""
        self._initialized = True

    def acquire(self, timeout=None, destroy_after_use=False):
        """Mock acquire context manager."""

        class MockContext:
            def __init__(self, manager):
                self.manager = manager
                self.instance = None

            def __enter__(self):
                self.instance = Mock()
                self.instance.model = MockSummarizationModel()
                self.instance.tokenizer = MockTokenizer()
                self.instance.device = torch.device("cpu")
                self.instance.model_type = "summarization"
                self.instance.in_use = True
                return self.instance

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.instance:
                    self.instance.in_use = False

        return MockContext(self)

    def summarize_text(self, text: str, instance=None) -> str:
        """Mock summarization method."""
        return f"Mock summary of: {text[:50]}..."

    def get_pool_stats(self):
        """Mock pool stats."""
        return {"available": 1, "in_use": 0, "total": 1, "max_instances": 2}


class MockEmbeddingModelManager:
    """Mock embedding model manager for testing."""

    def __init__(self, settings=None):
        self.settings = settings or Mock()
        self.settings.model_pool_max_instances = 2
        self._pool = None
        self._initialized = False

    def initialize(self):
        """Initialize the mock manager."""
        self._initialized = True

    def acquire(self, timeout=None, destroy_after_use=False):
        """Mock acquire context manager."""

        class MockContext:
            def __init__(self, manager):
                self.manager = manager
                self.instance = None

            def __enter__(self):
                self.instance = Mock()
                self.instance.model = MockEmbeddingModel()
                self.instance.tokenizer = None
                self.instance.device = torch.device("cpu")
                self.instance.model_type = "embedding"
                self.instance.in_use = True
                return self.instance

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.instance:
                    self.instance.in_use = False

        return MockContext(self)

    def encode_texts(self, texts, instance=None, batch_size=32, convert_to_tensor=True):
        """Mock encoding method."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = torch.randn(len(texts), 768)

        if convert_to_tensor:
            return embeddings
        else:
            return embeddings.tolist()

    def encode_single_text(self, text: str, instance=None):
        """Mock single text encoding."""
        return [0.1] * 768  # Mock embedding vector

    def compute_similarity(self, text1: str, text2: str, instance=None):
        """Mock similarity computation."""
        return 0.85  # Mock similarity score

    def get_pool_stats(self):
        """Mock pool stats."""
        return {"available": 1, "in_use": 0, "total": 1, "max_instances": 2}


class MockSettings:
    """Mock settings for testing."""

    def __init__(self, **kwargs):
        self.model_pool_max_instances = kwargs.get("model_pool_max_instances", 2)
        self.masx_force_cpu = kwargs.get("masx_force_cpu", True)
        self.masx_force_gpu = kwargs.get("masx_force_gpu", False)
        self.environment = kwargs.get("environment", "development")

        # GPU settings
        self.gpu_batch_size = kwargs.get("gpu_batch_size", 8)
        self.gpu_use_fp16 = kwargs.get("gpu_use_fp16", True)
        self.gpu_enable_warmup = kwargs.get("gpu_enable_warmup", True)

        # CPU settings
        self.cpu_batch_size = kwargs.get("cpu_batch_size", 2)
        self.cpu_max_threads = kwargs.get("cpu_max_threads", 4)
        self.cpu_max_processes = kwargs.get("cpu_max_processes", 2)


# Factory functions for easy testing
def create_mock_summarization_manager(settings=None):
    """Create a mock summarization model manager."""
    return MockSummarizationModelManager(settings)


def create_mock_embedding_manager(settings=None):
    """Create a mock embedding model manager."""
    return MockEmbeddingModelManager(settings)


def create_mock_settings(**kwargs):
    """Create mock settings with optional overrides."""
    return MockSettings(**kwargs)
