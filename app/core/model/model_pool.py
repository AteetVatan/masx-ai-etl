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

from typing import Dict, Set, Optional, TypeVar, Generic
import asyncio
from asyncio import Queue, QueueEmpty, QueueFull, Lock

from app.config import get_service_logger
from .model_Instance import ModelInstance

T = TypeVar("T")


class ModelPool(Generic[T]):
    """
    Async model pool with bounded capacity.

    - Uses asyncio.Queue for non-blocking acquire/release.
    - Maintains an 'in_use' set under an asyncio.Lock.
    - API:
        * async acquire(timeout: float | None) -> ModelInstance
        * async release(instance: ModelInstance, destroy: bool = False) -> None
        * add_instance(instance) -> bool        (sync, non-blocking)
        * get_stats() -> Dict[str, int]         (sync, best-effort)
        * shrink_pool(target_size: int) -> int  (sync, non-blocking)
    """

    def __init__(self, max_instances: int, model_type: str):
        self.max_instances = max_instances
        self.model_type = model_type
        self.available: Queue[ModelInstance[T]] = Queue(maxsize=max_instances)
        self._in_use: Set[ModelInstance[T]] = set()
        self._lock: Lock = Lock()
        self.logger = get_service_logger(f"ModelPool-{model_type}")

    async def acquire(self, timeout: Optional[float] = None) -> ModelInstance[T]:
        """
        Acquire a model instance asynchronously.

        Args:
            timeout: seconds to wait, None blocks indefinitely

        Raises:
            TimeoutError if waiting timed out.
        """
        try:
            if timeout is None:
                instance = await self.available.get()
            else:
                instance = await asyncio.wait_for(self.available.get(), timeout=timeout)

            async with self._lock:
                instance.in_use = True
                self._in_use.add(instance)

            self.logger.debug(f"Acquired {self.model_type} instance")
            return instance
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout waiting for {self.model_type} instance")
        except Exception as e:
            self.logger.error(f"Error acquiring {self.model_type} instance: {e}")
            raise e

    # method to release all instances
    async def release_all(self) -> None:
        """
        Release all model instances back to the pool.
        """
        async with self._lock:
            for instance in self._in_use:
                await self.release(instance)

    async def release(self, instance: ModelInstance[T], destroy: bool = False) -> None:
        """
        Release or destroy an instance asynchronously.
        """
        async with self._lock:
            if instance in self._in_use:
                self._in_use.remove(instance)
                instance.in_use = False

        if destroy:
            # Destruction may be blocking; if so, consider offloading to a thread.
            instance.destroy()
            self.logger.debug(f"Destroyed {self.model_type} instance")
            return

        try:
            # Non-blocking put; if pool is unexpectedly full, destroy instance
            self.available.put_nowait(instance)
            self.logger.debug(f"Released {self.model_type} instance to pool")
        except QueueFull:
            instance.destroy()
            self.logger.debug(f"Pool full, destroyed {self.model_type} instance")

    # --- Compatibility helpers (sync, best-effort) ---

    def add_instance(self, instance: ModelInstance[T]) -> bool:
        """
        Sync, non-blocking add during initialization.
        """
        try:
            self.available.put_nowait(instance)
            self.logger.debug(f"Added new {self.model_type} instance to pool")
            return True
        except QueueFull:
            return False

    def get_stats(self) -> Dict[str, int]:
        """
        Sync, best-effort stats (may be slightly stale without awaiting the lock).
        """
        return {
            "available": self.available.qsize(),
            "in_use": len(self._in_use),
            "total": self.available.qsize() + len(self._in_use),
            "max_instances": self.max_instances,
        }

    def shrink_pool(self, target_size: int) -> int:
        """
        Sync, non-blocking shrink by destroying idle instances above target.
        """
        destroyed = 0
        while self.available.qsize() > target_size:
            try:
                inst = self.available.get_nowait()
            except QueueEmpty:
                break
            inst.destroy()
            destroyed += 1

        if destroyed:
            self.logger.info(f"Shrunk {self.model_type} pool by {destroyed} instances")
        return destroyed
