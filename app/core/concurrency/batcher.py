"""
Micro-batching for GPU inference optimization.

This module provides efficient batching of inference requests to maximize GPU utilization
while maintaining bounded latency for real-time applications.
"""

import asyncio
import logging
import time
from typing import List, TypeVar, Callable, Awaitable, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import weakref

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchRequest:
    """Individual request in a batch."""

    id: str
    payload: T
    future: asyncio.Future
    timestamp: float
    priority: int = 0  # Higher = more urgent

    def __post_init__(self):
        """Set timestamp on creation."""
        if not hasattr(self, "timestamp"):
            self.timestamp = time.time()


class MicroBatcher:
    """
    Efficient micro-batching for GPU inference with configurable batching strategies.

    This class implements dynamic batching that balances throughput and latency:
    - Collects requests until batch size is reached OR max delay expires
    - Processes batches asynchronously to maximize GPU utilization
    - Provides backpressure and timeout handling
    """

    def __init__(
        self,
        batch_processor: Callable[[List[T]], Awaitable[List[R]]],
        max_batch_size: int = 32,
        max_delay_ms: int = 50,
        max_queue_size: int = 1000,
        timeout_seconds: float = 30.0,
    ):
        """
        Initialize the micro-batcher.

        Args:
            batch_processor: Async function that processes batches
            max_batch_size: Maximum items per batch
            max_delay_ms: Maximum delay before forcing batch processing (milliseconds)
            max_queue_size: Maximum pending requests in queue
            timeout_seconds: Request timeout in seconds
        """
        self.batch_processor = batch_processor
        self.max_batch_size = max_batch_size
        self.max_delay_ms = max_delay_ms
        self.max_queue_size = max_queue_size
        self.timeout_seconds = timeout_seconds

        # Internal state
        self._queue: deque[BatchRequest] = deque()
        self._current_batch: List[BatchRequest] = []
        self._batch_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()

        # Metrics
        self._total_requests = 0
        self._total_batches = 0
        self._total_processing_time = 0.0

        logger.info(
            f"MicroBatcher initialized: batch_size={max_batch_size}, "
            f"delay_ms={max_delay_ms}, queue_size={max_queue_size}"
        )

    async def submit(self, payload: T, request_id: Optional[str] = None) -> R:
        """
        Submit a request for processing.

        Args:
            payload: Data to process
            request_id: Optional request identifier

        Returns:
            Processing result

        Raises:
            asyncio.TimeoutError: If request times out
            RuntimeError: If batcher is shutdown
        """
        if self._shutdown_event.is_set():
            raise RuntimeError("MicroBatcher is shutdown")

        # Check queue capacity
        if len(self._queue) >= self.max_queue_size:
            raise RuntimeError(f"Queue full ({self.max_queue_size} requests)")

        # Create request
        request_id = request_id or f"req_{self._total_requests}"
        future = asyncio.Future()
        request = BatchRequest(
            id=request_id, payload=payload, future=future, timestamp=time.time()
        )

        async with self._lock:
            self._queue.append(request)
            self._total_requests += 1

            # Start batch processing if not already running
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._batch_loop())

        try:
            # Wait for result with timeout
            result = await asyncio.wait_for(future, timeout=self.timeout_seconds)
            return result
        except asyncio.TimeoutError:
            # Remove from queue if still there
            async with self._lock:
                if request in self._queue:
                    self._queue.remove(request)
            raise asyncio.TimeoutError(
                f"Request {request_id} timed out after {self.timeout_seconds}s"
            )

    async def _batch_loop(self):
        """Main batch processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for batch to be ready
                await self._wait_for_batch()

                # Process current batch
                if self._current_batch:
                    await self._process_batch()

            except asyncio.CancelledError:
                logger.debug("Batch loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch loop: {e}")
                # Continue processing other batches

    async def _wait_for_batch(self):
        """Wait until batch is ready for processing."""
        while not self._shutdown_event.is_set():
            async with self._lock:
                # Check if we have enough items for a batch
                if len(self._queue) >= self.max_batch_size:
                    self._current_batch = [
                        self._queue.popleft()
                        for _ in range(min(self.max_batch_size, len(self._queue)))
                    ]
                    return

                # Check if oldest item has exceeded max delay
                if self._queue and len(self._queue) > 0:
                    oldest_request = self._queue[0]
                    age_ms = (time.time() - oldest_request.timestamp) * 1000

                    if age_ms >= self.max_delay_ms:
                        # Take all available items
                        self._current_batch = [
                            self._queue.popleft() for _ in range(len(self._queue))
                        ]
                        return

            # Wait a bit before checking again
            await asyncio.sleep(0.001)  # 1ms

    async def _process_batch(self):
        """Process the current batch."""
        if not self._current_batch:
            return

        batch_size = len(self._current_batch)
        start_time = time.time()

        try:
            # Extract payloads
            payloads = [req.payload for req in self._current_batch]

            # Process batch
            results = await self.batch_processor(payloads)

            # Validate results
            if len(results) != batch_size:
                logger.error(
                    f"Batch processor returned {len(results)} results for {batch_size} requests"
                )
                # Set error for all requests
                for req in self._current_batch:
                    if not req.future.done():
                        req.future.set_exception(
                            RuntimeError(
                                f"Batch processor returned {len(results)} results for {batch_size} requests"
                            )
                        )
                return

            # Set results
            for req, result in zip(self._current_batch, results):
                if not req.future.done():
                    req.future.set_result(result)

            # Update metrics
            processing_time = time.time() - start_time
            self._total_batches += 1
            self._total_processing_time += processing_time

            logger.debug(
                f"Processed batch of {batch_size} items in {processing_time:.3f}s "
                f"(avg: {self._total_processing_time/self._total_batches:.3f}s)"
            )

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set error for all requests in batch
            for req in self._current_batch:
                if not req.future.done():
                    req.future.set_exception(e)

        finally:
            self._current_batch.clear()

    async def shutdown(self, wait: bool = True):
        """
        Shutdown the batcher gracefully.

        Args:
            wait: Whether to wait for pending batches to complete
        """
        logger.info("Shutting down MicroBatcher...")

        self._shutdown_event.set()

        # Cancel batch task
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
            if wait:
                try:
                    await self._batch_task
                except asyncio.CancelledError:
                    pass

        # Cancel all pending requests
        async with self._lock:
            for request in self._queue:
                if not request.future.done():
                    request.future.set_exception(RuntimeError("MicroBatcher shutdown"))
            self._queue.clear()

            for request in self._current_batch:
                if not request.future.done():
                    request.future.set_exception(RuntimeError("MicroBatcher shutdown"))
            self._current_batch.clear()

        logger.info("MicroBatcher shutdown complete")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "queue_size": len(self._queue),
            "current_batch_size": len(self._current_batch),
            "avg_processing_time": (
                self._total_processing_time / self._total_batches
                if self._total_batches > 0
                else 0.0
            ),
            "is_shutdown": self._shutdown_event.is_set(),
        }

    @property
    def queue_depth(self) -> int:
        """Current queue depth."""
        return len(self._queue)

    @property
    def is_shutdown(self) -> bool:
        """Check if batcher is shutdown."""
        return self._shutdown_event.is_set()
