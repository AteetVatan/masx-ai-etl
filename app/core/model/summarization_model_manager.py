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
Summarization model manager with pooling capabilities.

Manages BART summarization models with controlled concurrency and memory management.
"""

import os
import nltk
import torch
from typing import Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

from .abstract_model import AbstractModel, ModelInstance


class SummarizationModelManager(AbstractModel[AutoModelForSeq2SeqLM]):
    """
    Manager for BART summarization models with pooling capabilities.
    
    Provides controlled access to summarization model instances with automatic
    memory management and GPU/CPU optimization.
    """
    
    def __init__(self, settings: Optional[Any] = None):
        super().__init__(settings)
        self._model_name = "facebook/bart-large-cnn"
        self._max_tokens = 1024
        self._gen_config: Optional[GenerationConfig] = None
        
    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        return self._model_name
    
    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "summarization"
    
    @property
    def max_tokens(self) -> int:
        """Return the maximum number of tokens for summarization."""
        return self._max_tokens
    
    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens for summarization."""
        return self._max_tokens
    
    def get_tokenizer(self) -> AutoTokenizer:
        """Return the tokenizer for summarization."""
        return self._load_tokenizer()
    
    def get_device(self) -> torch.device:
        """Return the device for summarization."""
        return super().get_device()
    
    def get_model(self) -> AutoModelForSeq2SeqLM:
        """Return the model for summarization."""
        return self._load_model()
    
    def _get_model_vram_estimate(self) -> int:
        """Get estimated VRAM usage for BART-large model."""
        return 2 * 1024**3  # 2GB for BART-large    
    
    
    def _load_model(self) -> AutoModelForSeq2SeqLM:
        """Load and return a BART summarization model instance."""
        try:           
            device = self.get_device()
            use_fp16 = device.type == "cuda"
            
            # Load model with appropriate precision and memory optimization
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self._model_name,
                cache_dir=self._get_model_cache_dir(),
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
                low_cpu_mem_usage=True,
            )
            
            # Move to device and configure
            model = model.to(device)
            model.config.use_cache = True
            model.eval()
            
            if use_fp16:
                torch.set_float32_matmul_precision("high")
            
            self.logger.info(f"Loaded summarization model '{self._model_name}' on {device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load summarization model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load summarization model: {e}")
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load and return a BART tokenizer instance."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                cache_dir=self._get_model_cache_dir(),
                use_fast=True,
            )
            
            # Ensure pad token is set
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.logger.debug(f"Loaded tokenizer for {self._model_name}")
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    def _get_model_cache_dir(self) -> str:
        """Get the model cache directory."""        
        return self.model_cache_dir

    def get_generation_config(self) -> GenerationConfig:
        """Get the generation configuration for summarization."""
        if self._gen_config is None:
            self._gen_config = GenerationConfig(
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                do_sample=False,
            )
        return self._gen_config
    

    
    def summarize_text(self, text: str, instance: Optional[ModelInstance[AutoModelForSeq2SeqLM]] = None) -> str:
        """
        Summarize text using a model instance.
        
        Args:
            text: Text to summarize
            instance: Model instance to use (if None, will acquire one)
            
        Returns:
            Summarized text
        """
        should_release = instance is None
        
        if instance is None:
            with self.acquire() as instance:
                return self._perform_summarization(text, instance)
        else:
            return self._perform_summarization(text, instance)
    
    #discard this method
    def _perform_summarization(self, text: str, instance: ModelInstance[AutoModelForSeq2SeqLM]) -> str:
        """Perform the actual summarization using the model instance."""
        try:
            tokenizer = instance.tokenizer
            model = instance.model
            device = instance.device
            
            # Tokenize input
            inputs = tokenizer(
                text,
                max_length=self._max_tokens,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            
            gc=self.get_generation_config()
            n_words = len(text.split())
            gc.max_new_tokens = max(60, min(200, n_words // 2))   # ~50% of input words, clamped
            gc.min_new_tokens = min(40, gc.max_new_tokens // 3)
            
            # Generate summary
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=gc
                )
            
            # Decode output
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to summarize text: {e}")
            raise RuntimeError(f"Summarization failed: {e}")
