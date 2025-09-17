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
Unified inference runtime for MASX AI ETL.

This module provides the InferenceRuntime class that serves as the single entry point
for all inference operations, automatically selecting the optimal execution path
(GPU vs CPU) based on configuration and availability.
"""

import asyncio
import logging
import time
from typing import List, TypeVar, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass

from .device import use_gpu, get_device_config

# from .cpu_executors import CPUExecutors
# from .gpu_worker import GPUWorker, GPUConfig
from .model_pool import get_model_pool
from app.config import get_settings, get_service_logger
from app.core.model import AbstractModel

# from app.nlp import Translator, NLPUtils


T = TypeVar("T")
R = TypeVar("R")


@dataclass
class RuntimeConfig:
    """Runtime configuration parameters."""

    settings = get_settings()
    # Model Pool settings (for production mode)
    model_pool_max_instances: int = settings.model_pool_max_instances
    # Debug settings
    debug_mode: Optional[bool] = settings.debug
    # General settings
    enable_metrics: bool = settings.enable_metrics
    log_level: str = settings.log_level


class InferenceRuntime:
    """
    Unified inference runtime that automatically selects optimal execution path.

    This class provides a single API for inference operations that automatically:
    - Routes to GPU worker with micro-batching when GPU is available
    - Falls back to CPU executors (thread/process pools) when GPU is not available
    - Provides consistent async interface regardless of execution path
    - Handles model loading, batching, and resource management
    """

    def __init__(
        self,
        model_manager_loader: Optional[Callable] = None,
        config: Optional[RuntimeConfig] = None,
    ):
        """
        Initialize the inference runtime.

        Args:
            model_pool_loader: Function that loads and returns the model pool (for GPU path)
            config: Runtime configuration parameters
        """
        self.model_manager_loader = model_manager_loader
        self.model_manager: AbstractModel = None
        self.config = config or RuntimeConfig()
        self.logger = get_service_logger("InferenceRuntime")

        # Auto-detect debug mode if not specified
        if self.config.debug_mode is None:
            try:
                from app.config import get_settings

                settings = get_settings()
                self.config.debug_mode = settings.debug
            except Exception:
                self.config.debug_mode = False  # Default to production mode

        # Device configuration
        self.device_config = get_device_config()
        self.use_gpu_flag = use_gpu()

        # Execution components
        # self._gpu_worker: Optional[GPUWorker] = None
        # self._cpu_executors = CPUExecutors()

        # Model pool for production mode
        # self._model_pool = None
        # if not self.config.debug_mode and self.config.model_pool_enabled:
        #     self._model_pool = get_model_pool(self.config.model_pool_max_instances)

        # Runtime state
        self._is_started = False
        self._start_time = time.time()

        # Metrics
        self._total_requests = 0
        self._total_gpu_requests = 0
        self._total_cpu_requests = 0

        mode = "Debug" if self.config.debug_mode else "Production"
        # pool_info = (
        #     f", ModelPool={self._model_pool is not None}"
        #     if not self.config.debug_mode
        #     else ""
        # )

        # self.logger.info(
        #     f"runtime.py:InferenceRuntime initialized: Mode={mode}, GPU={self.use_gpu_flag}, "
        #     f"device={self.device_config.device_type}{pool_info}"
        # )

    async def start(self) -> None:
        """
        Start the inference runtime.

        This method initializes the appropriate execution path based on device configuration.
        """
        if self._is_started:
            self.logger.warning("runtime.py:InferenceRuntime already started")
            return

        try:
            # if self.use_gpu_flag and self.model_pool_loader:
            #     await self._start_gpu_worker()
            # else:
            #     self.logger.info("runtime.py:Using CPU execution path")

            self.model_manager = self.model_manager_loader()
            self.model_manager.initialize()

            self._is_started = True
            self.logger.info("runtime.py:InferenceRuntime started successfully")

        except Exception as e:
            self.logger.error(f"runtime.py:Failed to start InferenceRuntime: {e}")
            raise

    # async def stop(self) -> None:
    #     """
    #     Stop the inference runtime gracefully.
    #     """
    #     if self.model_manager:
    #         await self.model_manager.release_all_instances()
