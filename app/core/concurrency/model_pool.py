"""
Model Pool for controlled concurrency without memory explosion.

This module provides a ModelPool class that manages a limited number of model instances
to avoid memory explosion while still allowing some concurrency for inference operations.
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInstance:
    """Container for a model instance with metadata."""

    model: Any
    tokenizer: Any
    device: Any
    model_type: str
    in_use: bool = False
    created_at: float = 0.0


class ModelPool:
    """
    Thread-safe model pool with limited instances.

    This class manages a small pool of model instances to provide controlled
    concurrency without memory explosion. Each model type has its own pool
    with configurable limits.
    """

    def __init__(self, max_instances_per_type: int = 2):
        """
        Initialize the model pool.

        Args:
            max_instances_per_type: Maximum number of instances per model type
        """
        self.max_instances_per_type = max_instances_per_type
        self.pools: Dict[str, List[ModelInstance]] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.waiters: Dict[str, List[asyncio.Future]] = {}

        logger.info(
            f"ModelPool initialized with max {max_instances_per_type} instances per type"
        )

    async def get_model(self, model_type: str, model_loader_func) -> ModelInstance:
        """
        Get a model instance from the pool.

        Args:
            model_type: Type of model (e.g., 'summarization', 'embedding')
            model_loader_func: Function to load the model if not available

        Returns:
            ModelInstance: Available model instance
        """
        # Ensure we have a lock for this model type
        if model_type not in self.locks:
            self.locks[model_type] = asyncio.Lock()
            self.pools[model_type] = []
            self.waiters[model_type] = []

        async with self.locks[model_type]:
            # Try to find an available instance
            for instance in self.pools[model_type]:
                if not instance.in_use:
                    instance.in_use = True
                    logger.debug(f"Reusing {model_type} model instance")
                    return instance

            # If no available instance and we can create more
            if len(self.pools[model_type]) < self.max_instances_per_type:
                logger.info(
                    f"Creating new {model_type} model instance ({len(self.pools[model_type]) + 1}/{self.max_instances_per_type})"
                )

                try:
                    # Load the model
                    if model_type == "summarization":
                        model, tokenizer, device = model_loader_func()
                    elif model_type == "embedding":
                        model = model_loader_func()
                        tokenizer, device = None, None
                    else:
                        model, tokenizer, device = model_loader_func(), None, None

                    # Create instance
                    import time

                    instance = ModelInstance(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        model_type=model_type,
                        in_use=True,
                        created_at=time.time(),
                    )

                    self.pools[model_type].append(instance)
                    logger.info(f"Created {model_type} model instance successfully")
                    return instance

                except Exception as e:
                    logger.error(f"Failed to create {model_type} model instance: {e}")
                    raise

        # If we reach here, all instances are in use and we're at max capacity
        # Wait for an instance to become available
        logger.debug(f"All {model_type} model instances in use, waiting...")
        return await self._wait_for_available_instance(model_type)

    async def _wait_for_available_instance(self, model_type: str) -> ModelInstance:
        """Wait for an available model instance."""
        future = asyncio.Future()

        async with self.locks[model_type]:
            self.waiters[model_type].append(future)

        try:
            # Wait for notification that an instance is available
            await future

            # Try to get an instance again
            async with self.locks[model_type]:
                for instance in self.pools[model_type]:
                    if not instance.in_use:
                        instance.in_use = True
                        return instance

                # This shouldn't happen, but handle it gracefully
                raise RuntimeError(
                    f"No available {model_type} model instance after wait"
                )

        except asyncio.CancelledError:
            # Remove from waiters if cancelled
            async with self.locks[model_type]:
                if future in self.waiters[model_type]:
                    self.waiters[model_type].remove(future)
            raise

    async def return_model(self, instance: ModelInstance) -> None:
        """
        Return a model instance to the pool.

        Args:
            instance: The model instance to return
        """
        model_type = instance.model_type

        async with self.locks[model_type]:
            instance.in_use = False
            logger.debug(f"Returned {model_type} model instance to pool")

            # Notify waiting tasks
            if self.waiters[model_type]:
                waiter = self.waiters[model_type].pop(0)
                if not waiter.cancelled():
                    waiter.set_result(None)

    async def get_pool_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about the model pools."""
        stats = {}

        for model_type in self.pools:
            async with self.locks[model_type]:
                total = len(self.pools[model_type])
                in_use = sum(1 for inst in self.pools[model_type] if inst.in_use)
                available = total - in_use
                waiting = len(self.waiters[model_type])

                stats[model_type] = {
                    "total_instances": total,
                    "in_use": in_use,
                    "available": available,
                    "waiting_tasks": waiting,
                    "max_instances": self.max_instances_per_type,
                }

        return stats

    async def cleanup(self) -> None:
        """Clean up all model instances and clear pools."""
        logger.info("Cleaning up model pools...")

        for model_type in list(self.pools.keys()):
            async with self.locks[model_type]:
                # Cancel all waiters
                for waiter in self.waiters[model_type]:
                    if not waiter.cancelled():
                        waiter.cancel()

                # Clear instances (models will be garbage collected)
                self.pools[model_type].clear()
                self.waiters[model_type].clear()

                logger.info(f"Cleaned up {model_type} model pool")

        logger.info("Model pools cleanup complete")


# Global model pool instance
_global_model_pool: Optional[ModelPool] = None


def get_model_pool(max_instances_per_type: int = 2) -> ModelPool:
    """Get the global model pool instance."""
    global _global_model_pool

    if _global_model_pool is None:
        _global_model_pool = ModelPool(max_instances_per_type)

    return _global_model_pool


async def cleanup_model_pool():
    """Clean up the global model pool."""
    global _global_model_pool

    if _global_model_pool is not None:
        await _global_model_pool.cleanup()
        _global_model_pool = None
