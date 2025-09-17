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
Abstract base class for model managers with pooling capabilities.

This module provides the foundation for managing model instances with controlled
concurrency, memory management, and GPU/CPU resource optimization.
"""

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeVar, Generic
import asyncio
import torch

# from .model_pool import ModelPool_sync
from .model_pool import ModelPool
from .model_Instance import ModelInstance


from app.config import get_service_logger, get_settings

T = TypeVar("T")


class AbstractModel(ABC, Generic[T]):
    """
    Abstract base class for model managers with pooling capabilities.

    Provides the foundation for managing model instances with controlled
    concurrency, memory management, and GPU/CPU resource optimization.
    """

    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings or get_settings()
        self.model_cache_dir = self.settings.model_cache_dir
        self.logger = get_service_logger(self.__class__.__name__)
        self._pool: Optional[ModelPool[T]] = None
        self._pool_size: int = 0
        self._device: Optional[torch.device] = None
        self._initialized = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier."""
        pass

    @abstractmethod
    def _load_model(self) -> T:
        """Load and return a model instance."""
        pass

    @abstractmethod
    def _load_tokenizer(self) -> Optional[Any]:
        """Load and return a tokenizer instance (if applicable)."""
        pass

    @property
    def pool_size(self) -> int:
        """Get the pool size."""
        return self._pool_size

    def get_device(self) -> torch.device:
        """Get the appropriate device (GPU/CPU) for this model."""
        if self._device is None:
            try:
                from app.core.concurrency.device import get_torch_device

                self._device = get_torch_device()
                self.logger.info(f"Using device: {self._device}")
            except Exception as e:
                self.logger.error(f"Failed to get device: {e}")
                self._device = torch.device("cpu")
                self.logger.warning(f"Falling back to CPU device due to error: {e}")
        return self._device

    def _calculate_pool_size(self) -> int:
        """
        Calculate optimal pool size based on available resources.

        For GPU deployments, calculates based on available VRAM.
        For CPU deployments, uses settings value.
        """
        device = self.get_device()

        if device.type == "cuda":
            return self._calculate_gpu_pool_size()
        else:
            return self.settings.model_pool_max_instances

    def _calculate_gpu_pool_size(self) -> int:
        """
        Calculate pool size based on available GPU VRAM.

        Returns:
            Calculated pool size, clamped between 1 and model_pool_max_instances
        """
        try:
            # Try to get GPU memory info
            gpu_memory_info = self._get_gpu_memory_info()
            if gpu_memory_info:
                free_vram_bytes = gpu_memory_info.get("free_bytes", 0)
                model_vram_estimate = self._get_model_vram_estimate()

                if model_vram_estimate > 0:
                    max_instances = max(1, free_vram_bytes // model_vram_estimate)
                    pool_size = min(
                        max_instances, self.settings.model_pool_max_instances
                    )
                    self.logger.info(
                        f"GPU pool size calculated: {pool_size} (VRAM: {free_vram_bytes // 1024**3:.1f}GB, model estimate: {model_vram_estimate // 1024**3:.1f}GB)"
                    )
                    return pool_size
        except Exception as e:
            self.logger.warning(f"Failed to calculate GPU pool size: {e}")

        # Fallback to settings value
        return self.settings.model_pool_max_instances

    def _get_gpu_memory_info(self) -> Optional[Dict[str, int]]:
        """
        Get GPU memory information using nvidia-smi or pynvml.

        Returns:
            Dict with memory info or None if unavailable
        """
        try:
            # Try pynvml first
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    "total_bytes": mem_info.total,
                    "free_bytes": mem_info.free,
                    "used_bytes": mem_info.used,
                }
            except ImportError:
                pass
        except Exception:
            pass
            # Fallback to nvidia-smi
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.total,memory.free,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    total, free, used = map(int, result.stdout.strip().split(", "))
                    return {
                        "total_bytes": total * 1024 * 1024,  # Convert MB to bytes
                        "free_bytes": free * 1024 * 1024,
                        "used_bytes": used * 1024 * 1024,
                    }
            except Exception:
                pass

        return None

    def _get_model_vram_estimate(self) -> int:
        """
        Get estimated VRAM usage for this model type.

        Returns:
            Estimated VRAM usage in bytes
        """
        # Default estimates in bytes (can be overridden in subclasses)
        estimates = {
            "summarization": 2 * 1024**3,  # 2GB for BART-large
            "embedding": 1 * 1024**3,  # 1GB for sentence-transformers
            "translation": int(
                3.2 * 1024**3
            ),  # ~3.2 GB measured for NLLB-200-distilled-600M
        }
        return estimates.get(self.model_type, 2 * 1024**3)

    def initialize(self) -> None:
        """Initialize the model pool."""
        if self._initialized:
            return

        pool_size = self._calculate_pool_size()
        self._pool = ModelPool(pool_size, self.model_type)

        # Pre-populate pool with initial instances
        for _ in range(pool_size):
            self._create_and_add_instance()

        self._pool_size = pool_size
        self._initialized = True
        self.logger.info(
            f"Initialized {self.model_type} manager with pool size {pool_size}"
        )

    def _create_and_add_instance(self) -> bool:
        """Create a new model instance and add it to the pool."""
        try:
            model = self._load_model()
            tokenizer = self._load_tokenizer()
            device = self.get_device()

            instance = ModelInstance(
                model=model,
                tokenizer=tokenizer,
                device=device,
                model_type=self.model_type,
                created_at=time.time(),
                max_tokens=self.max_tokens,
                vram_usage_bytes=self._get_model_vram_estimate(),
            )

            return self._pool.add_instance(instance)
        except Exception as e:
            self.logger.error(f"Failed to create {self.model_type} instance: {e}")
            return False

    @asynccontextmanager
    async def acquire(
        self, timeout: Optional[float] = None, destroy_after_use: bool = False
    ):
        """
        Context manager for acquiring and releasing model instances.

        Args:
            timeout: Maximum time to wait for an instance
            destroy_after_use: If True, destroy the instance after use

        Yields:
            ModelInstance ready for use
        """
        if not self._initialized:
            self.initialize()

        instance = None
        try:
            instance = await self._pool.acquire(timeout=timeout)
            yield instance
        finally:
            if instance:
                await self._pool.release(instance, destroy=destroy_after_use)

    # method to release all instances
    async def release_all_instances(self) -> None:
        """
        Release all model instances back to the pool.
        """
        if self._pool:
            await self._pool.release_all()

    async def get_instance(self, timeout: Optional[float] = None) -> ModelInstance[T]:
        """
        Get a model instance (explicit acquire/release pattern).

        Args:
            timeout: Maximum time to wait for an instance

        Returns:
            ModelInstance ready for use
        """
        if not self._initialized:
            self.initialize()

        return await self._pool.acquire(timeout=timeout)

    def get_instance_sync(self, timeout: Optional[float] = None) -> ModelInstance[T]:
        """
        Get a model instance (explicit acquire/release pattern).

        Args:
            timeout: Maximum time to wait for an instance

        Returns:
            ModelInstance ready for use
        """
        if not self._initialized:
            self.initialize()

        return asyncio.run(self._pool.acquire(timeout=timeout))

    async def release_instance(
        self, instance: ModelInstance[T], destroy: bool = False
    ) -> None:
        """
        Release a model instance back to the pool.

        Args:
            instance: The instance to release
            destroy: If True, destroy the instance instead of returning to pool
        """
        if self._pool:
            await self._pool.release(instance, destroy=destroy)

    def release_instance_sync(
        self, instance: ModelInstance[T], destroy: bool = False
    ) -> None:
        """
        Release a model instance back to the pool.

        Args:
            instance: The instance to release
            destroy: If True, destroy the instance instead of returning to pool
        """
        if self._pool:
            asyncio.run(self._pool.release(instance, destroy=destroy))

    def get_pool_stats(self) -> Dict[str, int]:
        """Get current pool statistics."""
        if not self._pool:
            return {"error": "Pool not initialized"}
        return self._pool.get_stats()

    def shrink_pool(self, target_size: int) -> int:
        """
        Shrink the pool to target size.

        Args:
            target_size: Desired pool size

        Returns:
            Number of instances destroyed
        """
        if not self._pool:
            return 0
        return self._pool.shrink_pool(target_size)

    def cleanup(self) -> None:
        """Clean up all resources."""
        if self._pool:
            # Destroy all instances
            stats = self._pool.get_stats()
            self.shrink_pool(0)
            self.logger.info(
                f"Cleaned up {self.model_type} manager, destroyed {stats['total']} instances"
            )
