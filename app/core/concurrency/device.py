"""
Global device detection and configuration for MASX AI ETL.

This module provides a single source of truth for GPU detection and device selection,
eliminating scattered device checks across the codebase.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from app.config import get_settings

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    """Device configuration with GPU detection and fallback logic."""

    use_gpu: bool
    device_type: str
    device_id: Optional[int]
    cuda_available: bool
    force_cpu: bool
    force_gpu: bool

    def __post_init__(self):
        """Validate device configuration."""
        if self.force_gpu and not self.cuda_available:
            raise RuntimeError(
                "MASX_FORCE_GPU=1 but CUDA is not available. "
                "Please install CUDA-compatible PyTorch or set MASX_FORCE_CPU=1"
            )


def _detect_cuda() -> bool:
    """Detect CUDA availability safely."""
    if not TORCH_AVAILABLE:
        return False

    try:
        return torch.cuda.is_available()
    except Exception as e:
        logger.warning(f"device.py:CUDA detection failed: {e}")
        return False

def use_gpu() -> bool:
    """
    Determine if GPU should be used based on environment and availability.

    Returns:
        bool: True if GPU should be used, False otherwise

    Raises:
        RuntimeError: If MASX_FORCE_GPU=1 but CUDA unavailable
    """
    config = get_device_config()
    return config.use_gpu


def get_device_config() -> DeviceConfig:
    """
    Get the current device configuration.

    Returns:
        DeviceConfig: Complete device configuration

    Raises:
        RuntimeError: If MASX_FORCE_GPU=1 but CUDA unavailable
    """
    cuda_available = _detect_cuda()
    settings = get_settings()

    # Determine device selection logic
    if  settings.masx_force_cpu:
        use_gpu_flag = False
        device_type = "cpu"
        device_id = None
    elif settings.masx_force_gpu:
        if not cuda_available:
            raise RuntimeError(
                "MASX_FORCE_GPU=True but CUDA is not available. "
                "Please install CUDA-compatible PyTorch or set MASX_FORCE_CPU=1"
            )
        use_gpu_flag = True
        device_type = "cuda"
        device_id = 0
    else:
        # Auto-detect: use GPU if available
        use_gpu_flag = cuda_available
        device_type = "cuda" if cuda_available else "cpu"
        device_id = 0 if cuda_available else None

    config = DeviceConfig(
        use_gpu=use_gpu_flag,
        device_type=device_type,
        device_id=device_id,
        cuda_available=cuda_available,
        force_cpu=settings.masx_force_cpu,
        force_gpu=settings.masx_force_gpu,
    )

    logger.info(
        f"device.py:Device config: {config.device_type}"
        f"{f':{config.device_id}' if config.device_id is not None else ''}"
        f" (GPU: {config.use_gpu}, CUDA: {config.cuda_available})"
    )

    return config


def get_torch_device() -> torch.device:
    """
    Get PyTorch device object based on current configuration.

    Returns:
        torch.device: PyTorch device object

    Raises:
        RuntimeError: If PyTorch not available or device configuration invalid
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    config = get_device_config()

    # Handle device creation properly - torch.device() doesn't accept None as second argument
    if config.device_id is not None:
        device = torch.device(config.device_type, config.device_id)
        logger.info(
            f"device.py:Created PyTorch device: {device} (type: {config.device_type}, id: {config.device_id})"
        )
        return device
    else:
        device = torch.device(config.device_type)
        logger.info(
            f"device.py:Created PyTorch device: {device} (type: {config.device_type}, no id)"
        )
        return device


# Convenience functions for backward compatibility
def is_gpu_available() -> bool:
    """Check if GPU is available (deprecated, use use_gpu() instead)."""
    return use_gpu()


def get_device_string() -> str:
    """Get device string representation (deprecated, use get_device_config() instead)."""
    config = get_device_config()
    if config.device_id is not None:
        return f"{config.device_type}:{config.device_id}"
    return config.device_type
