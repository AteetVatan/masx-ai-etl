"""
Performance monitoring and optimization utilities for MASX AI ETL.

This module provides real-time performance monitoring, bottleneck detection,
and optimization recommendations for the concurrency system.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import threading
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring and optimization."""

    # Timing metrics
    start_time: float = field(default_factory=time.time)
    total_requests: int = 0
    total_batches: int = 0
    total_processing_time: float = 0.0

    # Throughput metrics
    requests_per_second: float = 0.0
    batches_per_second: float = 0.0
    avg_batch_time: float = 0.0

    # Resource utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: Optional[float] = None

    # Concurrency metrics
    active_workers: int = 0
    queue_depth: int = 0
    concurrent_batches: int = 0

    # Error metrics
    error_count: int = 0
    error_rate: float = 0.0

    # Optimization metrics
    batch_efficiency: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0


class PerformanceMonitor:
    """
    Real-time performance monitoring and optimization for the concurrency system.

    This class provides:
    - Real-time performance metrics collection
    - Bottleneck detection and analysis
    - Optimization recommendations
    - Resource utilization monitoring
    - Performance trend analysis
    """

    def __init__(self, update_interval: float = 1.0):
        """
        Initialize the performance monitor.

        Args:
            update_interval: Metrics update interval in seconds
        """
        self.update_interval = update_interval
        self.metrics = PerformanceMetrics()
        self.historical_metrics: deque = deque(
            maxlen=1000
        )  # Keep last 1000 measurements

        # Monitoring state
        self._is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # Performance thresholds for optimization
        self.performance_thresholds = {
            "gpu_utilization_min": 0.7,  # GPU should be at least 70% utilized
            "cpu_utilization_max": 0.8,  # CPU should not exceed 80%
            "batch_efficiency_min": 0.8,  # Batch efficiency should be at least 80%
            "error_rate_max": 0.05,  # Error rate should not exceed 5%
        }

        logger.info(
            f"performance_monitor.py:PerformanceMonitor initialized with {update_interval}s update interval"
        )

    async def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self._is_monitoring:
            logger.warning(
                "performance_monitor.py:Performance monitoring already started"
            )
            return

        self._is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("performance_monitor.py:Performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self._is_monitoring:
            return

        self._is_monitoring = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("performance_monitor.py:Performance monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for collecting performance metrics."""
        while self._is_monitoring:
            try:
                # Collect current metrics
                await self._collect_metrics()

                # Analyze performance and detect bottlenecks
                await self._analyze_performance()

                # Store historical data
                self._store_historical_metrics()

                # Wait for next update
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"performance_monitor.py:Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)

    async def _collect_metrics(self):
        """Collect current performance metrics."""
        try:
            with self._lock:
                # Update timing metrics
                current_time = time.time()
                uptime = current_time - self.metrics.start_time

                if uptime > 0:
                    self.metrics.requests_per_second = (
                        self.metrics.total_requests / uptime
                    )
                    self.metrics.batches_per_second = (
                        self.metrics.total_batches / uptime
                    )

                if self.metrics.total_batches > 0:
                    self.metrics.avg_batch_time = (
                        self.metrics.total_processing_time / self.metrics.total_batches
                    )

                # Update resource utilization
                self.metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
                self.metrics.memory_usage = psutil.virtual_memory().percent / 100.0

                # Try to get GPU usage if available
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.metrics.gpu_usage = gpu_util.gpu / 100.0
                except (ImportError, Exception):
                    self.metrics.gpu_usage = None

                # Update error rate
                if self.metrics.total_requests > 0:
                    self.metrics.error_rate = (
                        self.metrics.error_count / self.metrics.total_requests
                    )

                # Calculate efficiency metrics
                if self.metrics.total_batches > 0:
                    self.metrics.batch_efficiency = min(
                        1.0,
                        self.metrics.requests_per_second
                        / (self.metrics.total_batches * 10),
                    )

                if self.metrics.gpu_usage is not None:
                    self.metrics.gpu_utilization = self.metrics.gpu_usage

                self.metrics.cpu_utilization = self.metrics.cpu_usage / 100.0

        except Exception as e:
            logger.error(f"performance_monitor.py:Error collecting metrics: {e}")

    async def _analyze_performance(self):
        """Analyze performance and detect bottlenecks."""
        try:
            recommendations = []

            # Check GPU utilization
            if self.metrics.gpu_usage is not None:
                if (
                    self.metrics.gpu_usage
                    < self.performance_thresholds["gpu_utilization_min"]
                ):
                    recommendations.append(
                        {
                            "type": "gpu_utilization",
                            "severity": "warning",
                            "message": f'GPU utilization ({self.metrics.gpu_usage:.1%}) below threshold ({self.performance_thresholds["gpu_utilization_min"]:.1%})',
                            "suggestion": "Consider increasing batch size or reducing concurrent batches",
                        }
                    )

            # Check CPU utilization
            if (
                self.metrics.cpu_usage
                > self.performance_thresholds["cpu_utilization_max"] * 100
            ):
                recommendations.append(
                    {
                        "type": "cpu_utilization",
                        "severity": "warning",
                        "message": f'CPU utilization ({self.metrics.cpu_usage:.1%}) above threshold ({self.performance_thresholds["cpu_utilization_max"]:.1%})',
                        "suggestion": "Consider reducing concurrent CPU workers or increasing batch sizes",
                    }
                )

            # Check batch efficiency
            if (
                self.metrics.batch_efficiency
                < self.performance_thresholds["batch_efficiency_min"]
            ):
                recommendations.append(
                    {
                        "type": "batch_efficiency",
                        "severity": "info",
                        "message": f'Batch efficiency ({self.metrics.batch_efficiency:.1%}) below threshold ({self.performance_thresholds["batch_efficiency_min"]:.1%})',
                        "suggestion": "Consider optimizing batch sizes or reducing batch delays",
                    }
                )

            # Check error rate
            if self.metrics.error_rate > self.performance_thresholds["error_rate_max"]:
                recommendations.append(
                    {
                        "type": "error_rate",
                        "severity": "error",
                        "message": f'Error rate ({self.metrics.error_rate:.1%}) above threshold ({self.performance_thresholds["error_rate_max"]:.1%})',
                        "suggestion": "Investigate error sources and consider reducing concurrency",
                    }
                )

            # Log recommendations if any
            if recommendations:
                for rec in recommendations:
                    log_method = getattr(logger, rec["severity"], logger.info)
                    log_method(
                        f"performance_monitor.py:{rec['type'].upper()}: {rec['message']} - {rec['suggestion']}"
                    )

        except Exception as e:
            logger.error(f"performance_monitor.py:Error analyzing performance: {e}")

    def _store_historical_metrics(self):
        """Store current metrics in historical data."""
        try:
            with self._lock:
                # Create a copy of current metrics
                metrics_copy = PerformanceMetrics(
                    start_time=self.metrics.start_time,
                    total_requests=self.metrics.total_requests,
                    total_batches=self.metrics.total_batches,
                    total_processing_time=self.metrics.total_processing_time,
                    requests_per_second=self.metrics.requests_per_second,
                    batches_per_second=self.metrics.batches_per_second,
                    avg_batch_time=self.metrics.avg_batch_time,
                    cpu_usage=self.metrics.cpu_usage,
                    memory_usage=self.metrics.memory_usage,
                    gpu_usage=self.metrics.gpu_usage,
                    active_workers=self.metrics.active_workers,
                    queue_depth=self.metrics.queue_depth,
                    concurrent_batches=self.metrics.concurrent_batches,
                    error_count=self.metrics.error_count,
                    error_rate=self.metrics.error_rate,
                    batch_efficiency=self.metrics.batch_efficiency,
                    gpu_utilization=self.metrics.gpu_utilization,
                    cpu_utilization=self.metrics.cpu_utilization,
                )

                self.historical_metrics.append(metrics_copy)

        except Exception as e:
            logger.error(
                f"performance_monitor.py:Error storing historical metrics: {e}"
            )

    def record_request(self, processing_time: float = 0.0, is_error: bool = False):
        """Record a processed request."""
        try:
            with self._lock:
                self.metrics.total_requests += 1
                if processing_time > 0:
                    self.metrics.total_processing_time += processing_time
                if is_error:
                    self.metrics.error_count += 1

        except Exception as e:
            logger.error(f"performance_monitor.py:Error recording request: {e}")

    def record_batch(self, batch_size: int, processing_time: float):
        """Record a processed batch."""
        try:
            with self._lock:
                self.metrics.total_batches += 1
                if processing_time > 0:
                    self.metrics.total_processing_time += processing_time

        except Exception as e:
            logger.error(f"performance_monitor.py:Error recording batch: {e}")

    def update_concurrency_metrics(
        self, active_workers: int, queue_depth: int, concurrent_batches: int
    ):
        """Update concurrency-related metrics."""
        try:
            with self._lock:
                self.metrics.active_workers = active_workers
                self.metrics.queue_depth = queue_depth
                self.metrics.concurrent_batches = concurrent_batches

        except Exception as e:
            logger.error(
                f"performance_monitor.py:Error updating concurrency metrics: {e}"
            )

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            with self._lock:
                return {
                    "requests_per_second": self.metrics.requests_per_second,
                    "batches_per_second": self.metrics.batches_per_second,
                    "avg_batch_time": self.metrics.avg_batch_time,
                    "cpu_usage": self.metrics.cpu_usage,
                    "memory_usage": self.metrics.memory_usage,
                    "gpu_usage": self.metrics.gpu_usage,
                    "active_workers": self.metrics.active_workers,
                    "queue_depth": self.metrics.queue_depth,
                    "concurrent_batches": self.metrics.concurrent_batches,
                    "error_rate": self.metrics.error_rate,
                    "batch_efficiency": self.metrics.batch_efficiency,
                    "gpu_utilization": self.metrics.gpu_utilization,
                    "cpu_utilization": self.metrics.cpu_utilization,
                }
        except Exception as e:
            logger.error(f"performance_monitor.py:Error getting current metrics: {e}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        try:
            with self._lock:
                current_time = time.time()
                uptime = current_time - self.metrics.start_time

                # Calculate trends from historical data
                recent_metrics = (
                    list(self.historical_metrics)[-10:]
                    if len(self.historical_metrics) >= 10
                    else []
                )

                trend_analysis = {}
                if len(recent_metrics) >= 2:
                    first = recent_metrics[0]
                    last = recent_metrics[-1]

                    trend_analysis = {
                        "requests_per_second_trend": (
                            last.requests_per_second - first.requests_per_second
                        )
                        / max(first.requests_per_second, 1),
                        "error_rate_trend": (last.error_rate - first.error_rate)
                        / max(first.error_rate, 0.001),
                        "gpu_utilization_trend": (
                            (last.gpu_utilization - first.gpu_utilization)
                            if first.gpu_utilization and last.gpu_utilization
                            else 0
                        ),
                        "cpu_utilization_trend": (
                            last.cpu_utilization - first.cpu_utilization
                        )
                        / max(first.cpu_utilization, 0.001),
                    }

                return {
                    "uptime_seconds": uptime,
                    "total_requests": self.metrics.total_requests,
                    "total_batches": self.metrics.total_batches,
                    "current_metrics": self.get_current_metrics(),
                    "trend_analysis": trend_analysis,
                    "performance_score": self._calculate_performance_score(),
                    "optimization_recommendations": self._generate_optimization_recommendations(),
                }

        except Exception as e:
            logger.error(
                f"performance_monitor.py:Error getting performance summary: {e}"
            )
            return {}

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        try:
            score = 100.0

            # Deduct points for various issues
            if self.metrics.gpu_usage is not None and self.metrics.gpu_usage < 0.7:
                score -= (
                    0.7 - self.metrics.gpu_usage
                ) * 50  # Up to 35 points for GPU underutilization

            if self.metrics.cpu_usage > 80:
                score -= (
                    self.metrics.cpu_usage - 80
                ) * 0.5  # Up to 10 points for CPU overutilization

            if self.metrics.error_rate > 0.05:
                score -= (
                    self.metrics.error_rate * 200
                )  # Up to 10 points for high error rate

            if self.metrics.batch_efficiency < 0.8:
                score -= (
                    0.8 - self.metrics.batch_efficiency
                ) * 25  # Up to 20 points for low batch efficiency

            return max(0.0, min(100.0, score))

        except Exception as e:
            logger.error(
                f"performance_monitor.py:Error calculating performance score: {e}"
            )
            return 50.0  # Default middle score

    def _generate_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on current metrics."""
        recommendations = []

        try:
            # GPU optimization recommendations
            if self.metrics.gpu_usage is not None:
                if self.metrics.gpu_usage < 0.7:
                    recommendations.append(
                        {
                            "category": "GPU",
                            "priority": "high",
                            "action": "Increase batch size or reduce concurrent batches",
                            "reason": f"GPU utilization ({self.metrics.gpu_usage:.1%}) is below optimal threshold (70%)",
                        }
                    )

            # CPU optimization recommendations
            if self.metrics.cpu_usage > 80:
                recommendations.append(
                    {
                        "category": "CPU",
                        "priority": "medium",
                        "action": "Reduce concurrent CPU workers or increase batch sizes",
                        "reason": f"CPU utilization ({self.metrics.cpu_usage:.1%}) is above optimal threshold (80%)",
                    }
                )

            # Batch efficiency recommendations
            if self.metrics.batch_efficiency < 0.8:
                recommendations.append(
                    {
                        "category": "Batching",
                        "priority": "medium",
                        "action": "Optimize batch sizes and reduce batch delays",
                        "reason": f"Batch efficiency ({self.metrics.batch_efficiency:.1%}) is below optimal threshold (80%)",
                    }
                )

            # Error rate recommendations
            if self.metrics.error_rate > 0.05:
                recommendations.append(
                    {
                        "category": "Reliability",
                        "priority": "high",
                        "action": "Investigate error sources and consider reducing concurrency",
                        "reason": f"Error rate ({self.metrics.error_rate:.1%}) is above acceptable threshold (5%)",
                    }
                )

            # Queue depth recommendations
            if self.metrics.queue_depth > 100:
                recommendations.append(
                    {
                        "category": "Throughput",
                        "priority": "medium",
                        "action": "Increase processing capacity or optimize batch processing",
                        "reason": f"Queue depth ({self.metrics.queue_depth}) indicates processing bottleneck",
                    }
                )

        except Exception as e:
            logger.error(
                f"performance_monitor.py:Error generating optimization recommendations: {e}"
            )

        return recommendations

    async def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        try:
            collected = gc.collect()
            logger.info(
                f"performance_monitor.py:Forced garbage collection, collected {collected} objects"
            )
        except Exception as e:
            logger.error(f"performance_monitor.py:Error during garbage collection: {e}")

    def reset_metrics(self):
        """Reset all performance metrics."""
        try:
            with self._lock:
                self.metrics = PerformanceMetrics()
                self.historical_metrics.clear()
                logger.info("performance_monitor.py:Performance metrics reset")
        except Exception as e:
            logger.error(f"performance_monitor.py:Error resetting metrics: {e}")


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


async def start_performance_monitoring():
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    await monitor.start_monitoring()


async def stop_performance_monitoring():
    """Stop global performance monitoring."""
    monitor = get_performance_monitor()
    await monitor.stop_monitoring()
