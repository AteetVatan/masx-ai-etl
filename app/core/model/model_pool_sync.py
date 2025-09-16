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

from typing import Dict, List, Optional, TypeVar, Generic
import threading
from queue import Queue, Empty

from app.config import get_service_logger

from .model_Instance import ModelInstance

T = TypeVar('T')

class ModelPool_sync(Generic[T]):
    """
    Thread-safe model pool with limited instances and memory management.
    
    Provides controlled concurrency without memory explosion by managing
    a limited number of model instances per type.
    """
    
    def __init__(self, max_instances: int, model_type: str):
        self.max_instances = max_instances
        self.model_type = model_type
        self.available: Queue[ModelInstance[T]] = Queue(maxsize=max_instances)
        self.in_use: List[ModelInstance[T]] = []
        self.lock = threading.Lock()
        self.logger = get_service_logger(f"ModelPool-{model_type}")
        
    def acquire(self, timeout: Optional[float] = None) -> ModelInstance[T]:
        """
        Acquire a model instance from the pool.
        
        Args:
            timeout: Maximum time to wait for an instance (None = block forever)
            
        Returns:
            ModelInstance ready for use
            
        Raises:
            TimeoutError: If timeout is exceeded
        """
        try:
            instance = self.available.get(timeout=timeout)
            with self.lock:
                instance.in_use = True
                self.in_use.append(instance)
            self.logger.debug(f"Acquired {self.model_type} instance")
            return instance
        except Empty:
            raise TimeoutError(f"Timeout waiting for {self.model_type} instance")
    
    def release(self, instance: ModelInstance[T], destroy: bool = False) -> None:
        """
        Release a model instance back to the pool or destroy it.
        
        Args:
            instance: The instance to release
            destroy: If True, destroy the instance instead of returning to pool
        """
        with self.lock:
            if instance in self.in_use:
                self.in_use.remove(instance)
                instance.in_use = False
                
                if destroy:
                    instance.destroy()
                    self.logger.debug(f"Destroyed {self.model_type} instance")
                else:
                    try:
                        self.available.put_nowait(instance)
                        self.logger.debug(f"Released {self.model_type} instance to pool")
                    except:
                        # Pool is full, destroy the instance
                        instance.destroy()
                        self.logger.debug(f"Pool full, destroyed {self.model_type} instance")
    
    def add_instance(self, instance: ModelInstance[T]) -> bool:
        """
        Add a new instance to the pool.
        
        Args:
            instance: The instance to add
            
        Returns:
            True if added, False if pool is full
        """
        try:
            self.available.put_nowait(instance)
            self.logger.debug(f"Added new {self.model_type} instance to pool")
            return True
        except:
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self.lock:
            return {
                "available": self.available.qsize(),
                "in_use": len(self.in_use),
                "total": self.available.qsize() + len(self.in_use),
                "max_instances": self.max_instances
            }
    
    def shrink_pool(self, target_size: int) -> int:
        """
        Shrink the pool to target size by destroying excess instances.
        
        Args:
            target_size: Desired pool size
            
        Returns:
            Number of instances destroyed
        """
        destroyed = 0
        with self.lock:
            while self.available.qsize() > target_size:
                try:
                    instance = self.available.get_nowait()
                    instance.destroy()
                    destroyed += 1
                except Empty:
                    break
        
        if destroyed > 0:
            self.logger.info(f"Shrunk {self.model_type} pool by {destroyed} instances")
        return destroyed