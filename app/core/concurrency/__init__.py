"""
Core concurrency infrastructure for MASX AI ETL.

This package provides a unified, single-source concurrency runtime that eliminates
multithreading anti-patterns and provides optimal performance for both CPU and GPU workloads.
"""

from .device import use_gpu, get_device_config
from .runtime import InferenceRuntime
from .cpu_executors import CPUExecutors
from .gpu_worker import GPUWorker
from .batcher import MicroBatcher
from .runtime import RuntimeConfig
from .model_pool import ModelPool, get_model_pool
from .runpod_serverless_manager import RunPodServerlessManager

__all__ = [
    "use_gpu",
    "get_device_config",
    "InferenceRuntime",
    "CPUExecutors",
    "GPUWorker",
    "MicroBatcher",
    "RuntimeConfig",
    "ModelPool",
    "get_model_pool",
    "RunPodServerlessManager",
]
