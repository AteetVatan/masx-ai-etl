"""
Centralized CPU executors for MASX AI ETL.

This module provides shared thread and process pools to eliminate nested executor creation
and ensure optimal resource utilization across the application.
"""

import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional, Callable, Any, List, TypeVar, Awaitable
from functools import partial
import threading
from app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class CPUExecutors:
    """
    Centralized CPU executors with shared thread and process pools.

    This class ensures exactly one thread pool and one process pool per process,
    eliminating the anti-pattern of creating executors inside worker functions.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure one instance per process."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize executors only once."""
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._shutdown_event = threading.Event()
        self.settings = get_settings()

        # Configuration from environment
        self.max_threads = self.settings.cpu_max_threads
        self.max_processes = self.settings.cpu_max_processes

        logger.info(
            f"cpu_executors.py:CPUExecutors initialized: max_threads={self.max_threads}, max_processes={self.max_processes}"
        )

    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """Get or create the shared thread pool."""
        if self._thread_pool is None or self._thread_pool._shutdown:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.max_threads, thread_name_prefix="MASX-Thread"
            )
            logger.debug("cpu_executors.py:Created new thread pool")
        return self._thread_pool

    @property
    def process_pool(self) -> ProcessPoolExecutor:
        """Get or create the shared process pool."""
        if self._process_pool is None or self._process_pool._shutdown:
            self._process_pool = ProcessPoolExecutor(
                max_workers=self.max_processes,
                mp_context=None,  # Use default multiprocessing context
            )
            logger.debug("cpu_executors.py:Created new process pool")
        return self._process_pool

    async def run_in_thread(self, func: Callable[..., R], *args, **kwargs) -> R:
        """
        Run a blocking function in the shared thread pool.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: Any exception raised by the function
        """
        loop = asyncio.get_event_loop()
        executor = self.thread_pool

        try:
            result = await loop.run_in_executor(
                executor, partial(func, *args, **kwargs)
            )
            return result
        except Exception as e:
            logger.error(f"cpu_executors.py:Error in thread pool execution: {e}")
            raise

    async def run_in_process(self, func: Callable[..., R], *args, **kwargs) -> R:
        """
        Run a CPU-intensive function in the shared process pool.

        Args:
            func: Function to execute (must be picklable)
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: Any exception raised by the function
        """
        loop = asyncio.get_event_loop()
        executor = self.process_pool

        try:
            result = await loop.run_in_executor(
                executor, partial(func, *args, **kwargs)
            )
            return result
        except Exception as e:
            logger.error(f"cpu_executors.py:Error in process pool execution: {e}")
            raise

    async def map_threads(self, func: Callable[[T], R], items: List[T]) -> List[R]:
        """
        Map a function over items using the thread pool.

        Args:
            func: Function to apply to each item
            items: List of items to process

        Returns:
            List of results
        """
        tasks = [self.run_in_thread(func, item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def map_processes(self, func: Callable[[T], R], items: List[T]) -> List[R]:
        """
        Map a function over items using the process pool.

        Args:
            func: Function to apply to each item (must be picklable)
            items: List of items to process

        Returns:
            List of results
        """
        tasks = [self.run_in_process(func, item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def shutdown(self, wait: bool = True):
        """
        Shutdown all executors gracefully.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.info("cpu_executors.py:Shutting down CPU executors...")

        if self._thread_pool:
            self._thread_pool.shutdown(wait=wait)
            self._thread_pool = None

        if self._process_pool:
            self._process_pool.shutdown(wait=wait)
            self._process_pool = None

        self._shutdown_event.set()
        logger.info("cpu_executors.py:CPU executors shutdown complete")

    def is_shutdown(self) -> bool:
        """Check if executors are shutdown."""
        return self._shutdown_event.is_set()


# Global instance
cpu_executors = CPUExecutors()


# Convenience functions for easy access
async def run_in_thread(func: Callable[..., R], *args, **kwargs) -> R:
    """Run function in shared thread pool."""
    return await cpu_executors.run_in_thread(func, *args, **kwargs)


async def run_in_process(func: Callable[..., R], *args, **kwargs) -> R:
    """Run function in shared process pool."""
    return await cpu_executors.run_in_process(func, *args, **kwargs)


async def map_threads(func: Callable[[T], R], items: List[T]) -> List[R]:
    """Map function over items using thread pool."""
    return await cpu_executors.map_threads(func, items)


async def map_processes(func: Callable[[T], R], items: List[T]) -> List[R]:
    """Map function over items using process pool."""
    return await cpu_executors.map_processes(func, items)


def shutdown_executors(wait: bool = True):
    """Shutdown all executors."""
    cpu_executors.shutdown(wait=wait)
