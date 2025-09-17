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

import os
import torch
from typing import Any, Optional, List, Union
from sentence_transformers import SentenceTransformer
from .abstract_model import AbstractModel
from .model_Instance import ModelInstance
from transformers import AutoTokenizer


class VectorEmbeddingModelManager(AbstractModel[SentenceTransformer]):
    """
    Manager for SentenceTransformer embedding models with pooling capabilities.
    
    Provides controlled access to embedding model instances with automatic
    memory management and GPU/CPU optimization.
    """
    
    def __init__(self, settings: Optional[Any] = None):
        super().__init__(settings)
        self._model_name = "sentence-transformers/all-mpnet-base-v2"
        self._embedding_dimension = 768  # all-mpnet-base-v2 produces 768-dimensional embeddings
        
    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        return self._model_name
    
    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "embedding"
    
    @property
    def max_tokens(self) -> int:
        """Return the maximum number of tokens for embedding generation."""
        return 512  # SentenceTransformer default max sequence length
    
    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens for embedding generation."""
        return self.max_tokens
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for this model."""
        return self._embedding_dimension
    
    def get_device(self) -> torch.device:
        """Return the device for embedding generation."""
        return super().get_device()
    
    def get_model(self) -> SentenceTransformer:
        """Return the model for embedding generation."""
        return self._load_model()
    
    def _get_model_vram_estimate(self) -> int:
        """Get estimated VRAM usage for all-mpnet-base-v2 model."""
        return 1 * 1024**3  # 1GB for all-mpnet-base-v2
    
    def _load_model(self) -> SentenceTransformer:
        """Load and return a SentenceTransformer embedding model instance."""
        try:
            device = self.get_device()
            
            # Load model with appropriate device and memory optimization
            model = SentenceTransformer(
                self._model_name,
                cache_folder=self._get_model_cache_dir(),
                device=device,
            )
            
            # Configure model for optimal performance
            model.eval()
            
            # Set precision for GPU
            if device.type == "cuda":
                torch.set_float32_matmul_precision("high")
            
            self.logger.info(f"Loaded embedding model '{self._model_name}' on {device}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def _load_tokenizer(self) -> Optional[Any]:
        """Load and return a tokenizer for sentence-transformers/all-mpnet-base-v2."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,                # e.g. "sentence-transformers/all-mpnet-base-v2"
                cache_dir=self._get_model_cache_dir(),
                use_fast=True,
            )

            # Ensure pad token is set (MPNet has no pad by default)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    # Fall back to [PAD] if defined in vocab
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            self.logger.debug(f"Loaded tokenizer for {self._model_name} "
                            f"(pad_token={tokenizer.pad_token}, max_length={tokenizer.model_max_length})")
            return tokenizer

        except Exception as e:
            self.logger.error(f"Failed to load tokenizer for {self._model_name}: {e}")
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    def _get_model_cache_dir(self) -> str:
        """Get the model cache directory."""
        return self.model_cache_dir
    
    def encode_texts(
        self, 
        texts: Union[str, List[str]], 
        instance: Optional[ModelInstance[SentenceTransformer]] = None,
        batch_size: int = 32,
        convert_to_tensor: bool = False
    ) -> Union[List[List[float]], torch.Tensor]:
        """
        Encode texts into embeddings using a model instance.
        
        Args:
            texts: Text or list of texts to encode
            instance: Model instance to use (if None, will acquire one)
            batch_size: Batch size for encoding
            convert_to_tensor: Whether to return as tensor or list
            
        Returns:
            Embeddings as list of lists or tensor
        """
        should_release = instance is None
        
        if instance is None:
            with self.acquire() as instance:
                return self._perform_encoding(texts, instance, batch_size, convert_to_tensor)
        else:
            return self._perform_encoding(texts, instance, batch_size, convert_to_tensor)
    
    def encode_text(self, text: str, instance: Optional[ModelInstance[SentenceTransformer]] = None) -> List[float]:
        """
        Encode a single text to embedding using a model instance.
        
        Args:
            text: Text to encode
            instance: Model instance to use (if None, will acquire one)
            
        Returns:
            Embedding vector as list of floats
        """
        embeddings = self.encode_texts([text], instance, convert_to_tensor=False)
        return embeddings[0]
    
    def _perform_encoding(
        self, 
        texts: Union[str, List[str]], 
        instance: ModelInstance[SentenceTransformer],
        batch_size: int,
        convert_to_tensor: bool
    ) -> Union[List[List[float]], torch.Tensor]:
        """Perform the actual encoding using the model instance."""
        try:
            model = instance.model
            
            # Encode texts
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=convert_to_tensor,
                show_progress_bar=False
            )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to encode texts: {e}")
            raise RuntimeError(f"Text encoding failed: {e}")
    
    def compute_similarity(
        self, 
        text1: str, 
        text2: str, 
        instance: Optional[ModelInstance[SentenceTransformer]] = None
    ) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            instance: Model instance to use (if None, will acquire one)
            
        Returns:
            Similarity score between 0 and 1
        """
        should_release = instance is None
        
        if instance is None:
            with self.acquire() as instance:
                return self._compute_similarity(text1, text2, instance)
        else:
            return self._compute_similarity(text1, text2, instance)
    
    def _compute_similarity(
        self, 
        text1: str, 
        text2: str, 
        instance: ModelInstance[SentenceTransformer]
    ) -> float:
        """Compute similarity between two texts using the model instance."""
        try:
            model = instance.model
            
            # Encode both texts
            embeddings = model.encode([text1, text2], convert_to_tensor=True)
            
            # Compute cosine similarity
            similarity = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
            return float(similarity.item())
            
        except Exception as e:
            self.logger.error(f"Failed to compute similarity: {e}")
            raise RuntimeError(f"Similarity computation failed: {e}")
