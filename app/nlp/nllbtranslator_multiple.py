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
High-performance GPU-optimized NLLB-200 multilingual translation system.

This module provides:
- Multiple GPU model instances for parallel processing
- Efficient GPU memory management
- Batch processing optimization
- Automatic GPU device selection and load balancing
- Fallback to CPU when GPU resources are exhausted
"""

import asyncio
import logging
import time
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from collections import deque
import threading
import weakref

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.config import get_settings, get_service_logger


@dataclass
class TranslationModelInstance:
    """Individual translation model instance with GPU management."""
    
    model_id: str
    model: Any
    tokenizer: Any
    device: str
    src_lang: str
    tgt_lang: str
    last_used: float
    is_available: bool = True
    memory_usage_mb: float = 0.0
    
    def __post_init__(self):
        """Set last_used timestamp on creation."""
        if not hasattr(self, 'last_used'):
            self.last_used = time.time()


class NLLBTranslator_multiple:
    """
    High-performance GPU-optimized translation manager.
    
    This class manages multiple translation model instances using existing
    GPU/CPU settings and provides a model pool for efficient resource management.
    """
    
    def __init__(self):
        """Initialize the translation manager."""
        self.settings = get_settings()
        self.logger = get_service_logger("NLLBTranslator")
        
        # Model configuration
        self.model_name = "facebook/nllb-200-distilled-600M"
        
        # Use existing GPU/CPU settings based on environment
        self._configure_device_settings()
        
        # Model instance management with pool
        self._gpu_instances: Dict[str, TranslationModelInstance] = {}
        self._cpu_instances: Dict[str, TranslationModelInstance] = {}
        self._instance_lock = threading.Lock()
        
        # Performance metrics
        self._total_translations = 0
        self._total_batches = 0
        self._start_time = time.time()
        
        # GPU device management
        self._available_gpus = self._detect_available_gpus()
        self._gpu_load_balancer = self._create_gpu_load_balancer()
        
        self.logger.info(
            f"nllbtranslator.py:NLLBTranslator initialized with "
            f"{len(self._available_gpus)} GPU devices, max instances: {self.max_gpu_instances}"
        )
    
    def _configure_device_settings(self):
        """Configure device settings based on existing environment variables."""
        try:
            # Check environment settings for device preference
            if self.settings.masx_force_gpu:
                # Use existing GPU settings
                self.max_gpu_instances = self.settings.model_pool_max_instances
                self.max_cpu_instances = 0  # No CPU fallback when forcing GPU
                
            elif self.settings.masx_force_cpu:
                # Use existing CPU settings
                self.max_gpu_instances = 0  # No GPU when forcing CPU
                self.max_cpu_instances = self.settings.model_pool_max_instances
                
                # CPU configuration from existing settings
                self.cpu_batch_size = self.settings.cpu_batch_size

                
            else:
                # Auto-detect: prefer GPU if available, fallback to CPU
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    # GPU preferred
                    self.max_gpu_instances = self.settings.model_pool_max_instances
                    self.max_cpu_instances = 2  # Limited CPU fallback
                    

                    
                else:
                    # CPU only
                    self.max_gpu_instances = 0
                    self.max_cpu_instances = self.settings.model_pool_max_instances
                    
                    # CPU configuration
                    self.cpu_batch_size = self.settings.cpu_batch_size

            
            # Model pool configuration
            self.model_pool_enabled = self.settings.nllb_model_pool_enabled
            self.model_pool_max_instances = self.settings.nllb_model_pool_max_instances
            self.model_pool_timeout = self.settings.nllb_model_pool_timeout
            self.model_pool_cleanup_interval = self.settings.nllb_model_pool_cleanup_interval
            
        except Exception as e:
            self.logger.warning(f"Error configuring device settings, using defaults: {e}")
            # Fallback to safe defaults
            self.max_gpu_instances = 2
            self.max_cpu_instances = 2
            self.gpu_batch_size = 32
            self.cpu_batch_size = 16
    
    def _get_max_gpu_instances(self) -> int:
        """Calculate optimal number of GPU instances based on available memory."""
        try:
            if not TORCH_AVAILABLE or not torch.cuda.is_available():
                return 0
            
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            # NLLB-200 model requires ~1.2GB GPU memory
            # Reserve 20% for system overhead
            available_memory = gpu_memory * 0.8
            max_instances = int(available_memory / 1.2)
            
            # Cap at reasonable limits and respect model pool setting
            max_instances = min(max_instances, self.model_pool_max_instances)
            max_instances = max(max_instances, 1)  # At least 1 instance
            
            self.logger.info(f"GPU memory: {gpu_memory:.1f}GB, max instances: {max_instances}")
            return max_instances
            
        except Exception as e:
            self.logger.warning(f"Error calculating GPU instances, using default: {e}")
            return min(2, self.model_pool_max_instances)  # Safe default
    
    def _detect_available_gpus(self) -> List[int]:
        """Detect available GPU devices."""
        try:
            if not TORCH_AVAILABLE or not torch.cuda.is_available():
                return []
            
            gpu_count = torch.cuda.device_count()
            available_gpus = []
            
            for i in range(gpu_count):
                try:
                    # Check if GPU is accessible
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    available_gpus.append(i)
                except Exception as e:
                    self.logger.warning(f"GPU {i} not accessible: {e}")
            
            self.logger.info(f"Detected {len(available_gpus)} available GPU devices: {available_gpus}")
            return available_gpus
            
        except Exception as e:
            self.logger.error(f"Error detecting GPUs: {e}")
            return []
    
    def _create_gpu_load_balancer(self) -> Dict[int, int]:
        """Create GPU load balancer for distributing instances."""
        load_balancer = {}
        if self._available_gpus:
            instances_per_gpu = max(1, self.max_gpu_instances // len(self._available_gpus))
            for gpu_id in self._available_gpus:
                load_balancer[gpu_id] = instances_per_gpu
        
        self.logger.info(f"GPU load balancer: {load_balancer}")
        return load_balancer
    
    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text with automatic GPU/CPU instance management.
        
        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            
        Returns:
            Translated text
        """
        try:
            # Get or create translation instance
            instance = await self._get_translation_instance(src_lang, tgt_lang)
            if not instance:
                self.logger.error("Failed to get translation instance")
                return text  # Return original text on failure
            
            # Perform translation
            start_time = time.time()
            result = await self._translate_with_instance(instance, text)
            processing_time = time.time() - start_time
            
            # Update metrics
            self._total_translations += 1
            
            # Release instance back to pool
            await self._release_instance(instance)
            
            self.logger.debug(
                f"Translation completed in {processing_time:.3f}s "
                f"using {instance.device} (total: {self._total_translations})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return text  # Return original text on failure
    
    async def translate_batch(
        self, texts: List[str], src_lang: str, tgt_lang: str
    ) -> List[str]:
        """
        Translate batch of texts with optimized GPU processing.
        
        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        try:
            # Get multiple instances for parallel processing
            instances = await self._get_multiple_instances(src_lang, tgt_lang, len(texts))
            if not instances:
                self.logger.error("Failed to get translation instances")
                return texts  # Return original texts on failure
            
            # Process texts in parallel using available instances
            start_time = time.time()
            results = await self._translate_batch_parallel(texts, instances, src_lang, tgt_lang)
            processing_time = time.time() - start_time
            
            # Update metrics
            self._total_translations += len(texts)
            self._total_batches += 1
            
            # Release all instances back to pool
            for instance in instances:
                await self._release_instance(instance)
            
            self.logger.info(
                f"Batch translation completed: {len(texts)} texts in {processing_time:.3f}s "
                f"using {len(instances)} instances (total: {self._total_translations})"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch translation failed: {e}")
            return texts  # Return original texts on failure
    
    async def _get_translation_instance(self, src_lang: str, tgt_lang: str) -> Optional[TranslationModelInstance]:
        """Get or create a translation instance from the model pool."""
        try:
            instance_key = f"{src_lang}->{tgt_lang}"
            
            # Check existing instances first
            with self._instance_lock:
                # Look for available GPU instance
                for instance in self._gpu_instances.values():
                    if (instance.src_lang == src_lang and 
                        instance.tgt_lang == tgt_lang and 
                        instance.is_available):
                        instance.is_available = False
                        instance.last_used = time.time()
                        return instance
                
                # Look for available CPU instance
                for instance in self._cpu_instances.values():
                    if (instance.src_lang == src_lang and 
                        instance.tgt_lang == tgt_lang and 
                        instance.is_available):
                        instance.is_available = False
                        instance.last_used = time.time()
                        return instance
            
            # Create new instance if needed and pool allows
            if self._can_create_new_instance():
                return await self._create_translation_instance(src_lang, tgt_lang)
            else:
                # Wait for an instance to become available
                return await self._wait_for_available_instance(src_lang, tgt_lang)
            
        except Exception as e:
            self.logger.error(f"Error getting translation instance: {e}")
            return None
    
    def _can_create_new_instance(self) -> bool:
        """Check if we can create a new instance within pool limits."""
        total_instances = len(self._gpu_instances) + len(self._cpu_instances)
        return total_instances < self.model_pool_max_instances
    
    async def _wait_for_available_instance(self, src_lang: str, tgt_lang: str) -> Optional[TranslationModelInstance]:
        """Wait for an instance to become available."""
        max_wait_time = self.model_pool_timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Check for available instances
            with self._instance_lock:
                for instance in self._gpu_instances.values():
                    if (instance.src_lang == src_lang and 
                        instance.tgt_lang == tgt_lang and 
                        instance.is_available):
                        instance.is_available = False
                        instance.last_used = time.time()
                        return instance
                
                for instance in self._cpu_instances.values():
                    if (instance.src_lang == src_lang and 
                        instance.tgt_lang == tgt_lang and 
                        instance.is_available):
                        instance.is_available = False
                        instance.last_used = time.time()
                        return instance
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        self.logger.warning(f"Timeout waiting for available instance for {src_lang}->{tgt_lang}")
        return None
    
    async def _get_multiple_instances(
        self, src_lang: str, tgt_lang: str, text_count: int
    ) -> List[TranslationModelInstance]:
        """Get multiple instances for parallel batch processing."""
        try:
            instances = []
            instance_key = f"{src_lang}->{tgt_lang}"
            
            # Calculate optimal number of instances
            optimal_instances = min(
                text_count,
                self.max_gpu_instances + self.max_cpu_instances,
                4  # Cap at 4 instances for batch processing
            )
            
            # Get existing available instances
            with self._instance_lock:
                # GPU instances first
                for instance in self._gpu_instances.values():
                    if (instance.src_lang == src_lang and 
                        instance.tgt_lang == tgt_lang and 
                        instance.is_available and
                        len(instances) < optimal_instances):
                        instance.is_available = False
                        instance.last_used = time.time()
                        instances.append(instance)
                
                # CPU instances as fallback
                for instance in self._cpu_instances.values():
                    if (instance.src_lang == src_lang and 
                        instance.tgt_lang == tgt_lang and 
                        instance.is_available and
                        len(instances) < optimal_instances):
                        instance.is_available = False
                        instance.last_used = time.time()
                        instances.append(instance)
            
            # Create additional instances if needed
            while len(instances) < optimal_instances:
                new_instance = await self._create_translation_instance(src_lang, tgt_lang)
                if new_instance:
                    instances.append(new_instance)
                else:
                    break
            
            self.logger.info(f"Got {len(instances)} instances for batch processing")
            return instances
            
        except Exception as e:
            self.logger.error(f"Error getting multiple instances: {e}")
            return []
    
    async def _create_translation_instance(self, src_lang: str, tgt_lang: str) -> Optional[TranslationModelInstance]:
        """Create a new translation instance."""
        try:
            instance_key = f"{src_lang}->{tgt_lang}"
            
            # Determine device (GPU or CPU)
            device, device_type = self._select_device()
            
            # Load model and tokenizer
            model, tokenizer = await self._load_model_components()
            if not model or not tokenizer:
                return None
            
            # Move model to selected device
            if device_type == "gpu":
                model = model.to(device)
                # Clear GPU cache to prevent memory issues
                torch.cuda.empty_cache()
            
            # Create instance
            instance = TranslationModelInstance(
                model_id=f"{instance_key}_{len(self._gpu_instances) + len(self._cpu_instances)}",
                model=model,
                tokenizer=tokenizer,
                device=device,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                last_used=time.time(),
                is_available=False  # Mark as in use
            )
            
            # Store instance in appropriate pool
            with self._instance_lock:
                if device_type == "gpu":
                    self._gpu_instances[instance.model_id] = instance
                else:
                    self._cpu_instances[instance.model_id] = instance
            
            self.logger.info(
                f"Created new {device_type.upper()} translation instance: {instance.model_id} "
                f"on device {device}"
            )
            
            return instance
            
        except Exception as e:
            self.logger.error(f"Error creating translation instance: {e}")
            return None
    
    def _select_device(self) -> Tuple[str, str]:
        """Select optimal device for new instance."""
        try:
            # Check if we can create more GPU instances
            if (self._available_gpus and 
                len(self._gpu_instances) < self.max_gpu_instances):
                
                # Select GPU with least load
                gpu_loads = {}
                for gpu_id in self._available_gpus:
                    gpu_loads[gpu_id] = sum(
                        1 for instance in self._gpu_instances.values()
                        if instance.device == f"cuda:{gpu_id}"
                    )
                
                # Find GPU with minimum load
                selected_gpu = min(gpu_loads.keys(), key=lambda x: gpu_loads[x])
                return f"cuda:{selected_gpu}", "gpu"
            
            # Fallback to CPU
            return "cpu", "cpu"
            
        except Exception as e:
            self.logger.warning(f"Error selecting device, using CPU: {e}")
            return "cpu", "cpu"
    
    async def _load_model_components(self) -> Tuple[Any, Any]:
        """Load model and tokenizer components."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model components: {e}")
            return None, None
    
    async def _translate_with_instance(self, instance: TranslationModelInstance, text: str) -> str:
        """Translate text using a specific instance."""
        try:
            # Create translation pipeline
            from transformers import pipeline
            
            translator = pipeline(
                "translation",
                model=instance.model,
                tokenizer=instance.tokenizer,
                src_lang=instance.src_lang,
                tgt_lang=instance.tgt_lang,
                max_length=512,
                device=instance.device,
            )
            
            # Perform translation
            result = translator(text)
            return result[0]["translation_text"]
            
        except Exception as e:
            self.logger.error(f"Translation with instance failed: {e}")
            return text  # Return original text on failure
    
    async def _translate_batch_parallel(
        self, texts: List[str], instances: List[TranslationModelInstance], 
        src_lang: str, tgt_lang: str
    ) -> List[str]:
        """Translate batch of texts using multiple instances in parallel."""
        try:
            if not instances:
                return texts
            
            # Split texts among instances
            texts_per_instance = len(texts) // len(instances)
            remainder = len(texts) % len(instances)
            
            # Create tasks for parallel processing
            tasks = []
            text_index = 0
            
            for i, instance in enumerate(instances):
                # Calculate texts for this instance
                instance_text_count = texts_per_instance + (1 if i < remainder else 0)
                instance_texts = texts[text_index:text_index + instance_text_count]
                text_index += instance_text_count
                
                if instance_texts:
                    task = self._translate_texts_with_instance(instance, instance_texts)
                    tasks.append(task)
            
            # Execute all tasks in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Flatten and order results
                all_results = []
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Batch translation task failed: {result}")
                        # Fill with original texts on error
                        all_results.extend([texts[i] for i in range(len(all_results), len(all_results) + len(result))])
                    else:
                        all_results.extend(result)
                
                return all_results[:len(texts)]  # Ensure correct length
            
            return texts
            
        except Exception as e:
            self.logger.error(f"Parallel batch translation failed: {e}")
            return texts
    
    async def _translate_texts_with_instance(
        self, instance: TranslationModelInstance, texts: List[str]
    ) -> List[str]:
        """Translate multiple texts using a single instance."""
        try:
            # Create translation pipeline
            from transformers import pipeline
            
            translator = pipeline(
                "translation",
                model=instance.model,
                tokenizer=instance.tokenizer,
                src_lang=instance.src_lang,
                tgt_lang=instance.tgt_lang,
                max_length=512,
                device=instance.device,
            )
            
            # Process texts in small batches to avoid memory issues
            batch_size = self.gpu_batch_size if instance.device.startswith("cuda") else self.cpu_batch_size
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    batch_results = translator(batch)
                    batch_translations = [result["translation_text"] for result in batch_results]
                    results.extend(batch_translations)
                except Exception as e:
                    self.logger.error(f"Batch translation failed: {e}")
                    # Fill with original texts on error
                    results.extend(batch)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Text translation with instance failed: {e}")
            return texts  # Return original texts on failure
    
    async def _release_instance(self, instance: TranslationModelInstance):
        """Release instance back to the pool."""
        try:
            with self._instance_lock:
                instance.is_available = True
                instance.last_used = time.time()
            
            self.logger.debug(f"Released instance: {instance.model_id}")
            
        except Exception as e:
            self.logger.error(f"Error releasing instance: {e}")
    
    async def cleanup(self):
        """Clean up resources and clear GPU memory."""
        try:
            with self._instance_lock:
                # Clear all instances
                for instance in self._gpu_instances.values():
                    if hasattr(instance.model, 'cpu'):
                        instance.model.cpu()
                    del instance.model
                
                for instance in self._cpu_instances.values():
                    del instance.model
                
                self._gpu_instances.clear()
                self._cpu_instances.clear()
            
            # Clear GPU cache
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Translation manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get translation manager metrics."""
        uptime = time.time() - self._start_time
        
        return {
            'uptime_seconds': uptime,
            'total_translations': self._total_translations,
            'total_batches': self._total_batches,
            'translations_per_second': self._total_translations / uptime if uptime > 0 else 0,
            'gpu_instances': len(self._gpu_instances),
            'cpu_instances': len(self._cpu_instances),
            'available_gpus': len(self._available_gpus),
            'max_gpu_instances': self.max_gpu_instances,
            'max_cpu_instances': self.max_cpu_instances,
            'model_pool_enabled': self.model_pool_enabled,
            'model_pool_max_instances': self.model_pool_max_instances,
            'model_pool_timeout': self.model_pool_timeout,
            'model_pool_cleanup_interval': self.model_pool_cleanup_interval
        }
    
    def get_instance_status(self) -> Dict[str, Any]:
        """Get detailed instance status information."""
        try:
            gpu_status = {}
            cpu_status = {}
            
            with self._instance_lock:
                for instance_id, instance in self._gpu_instances.items():
                    gpu_status[instance_id] = {
                        'device': instance.device,
                        'src_lang': instance.src_lang,
                        'tgt_lang': instance.tgt_lang,
                        'is_available': instance.is_available,
                        'last_used': instance.last_used,
                        'memory_usage_mb': instance.memory_usage_mb
                    }
                
                for instance_id, instance in self._cpu_instances.items():
                    cpu_status[instance_id] = {
                        'device': instance.device,
                        'src_lang': instance.src_lang,
                        'tgt_lang': instance.tgt_lang,
                        'is_available': instance.is_available,
                        'last_used': instance.last_used
                    }
            
            return {
                'gpu_instances': gpu_status,
                'cpu_instances': cpu_status,
                'total_instances': len(gpu_status) + len(cpu_status)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting instance status: {e}")
            return {}

# Global instance of the NLLBTranslator
nllb_translator_instance = NLLBTranslator_multiple()


def get_nllb_translator() -> NLLBTranslator_multiple:
    """Get the global NLLB translator instance."""
    return nllb_translator_instance


async def cleanup_nllb_translator():
    """Clean up the global NLLB translator instance."""
    await nllb_translator_instance.cleanup()