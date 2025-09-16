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
This module contains the Model class, which is a singleton class that loads and manages the models.
"""

import os
import threading


import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime, timedelta
from app.services import ProxyService


from app.config import get_service_logger

# local directory path where Hugging Face will store the downloaded model weights, tokenizers, and configuration files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE = os.path.join(BASE_DIR, "..", ".hf_cache")  # or any central path


class ModelManager_old:
    """Singleton class to load and manage models and translators with CPU/GPU detection."""

    _summarization_model: AutoModelForSeq2SeqLM | None = None
    _summarization_tokenizer: AutoTokenizer | None = None
    _device: torch.device | None = None
    _summarization_model_max_tokens: int = 1024

    _summarization_model_name: str = "facebook/bart-large-cnn"
    _embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"

    _embedding_model: SentenceTransformer | None = None
    _embedding_lock = threading.Lock()
    
  

    _logger = get_service_logger("ModelManager")

    # ===== PATH HELPERS =====
    @classmethod
    def get_base_dir(cls) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def get_model_cache_dir(cls) -> str:
        return os.path.join(cls.get_base_dir(), "..", ".hf_cache")

    @classmethod
    def get_fasttext_model_path(cls) -> str:
        return os.path.join(cls.get_base_dir(), "..", "lid.176.bin")

    @classmethod
    def get_summarization_model_max_tokens(cls) -> int:
        """Get the maximum number of tokens for the summarization model."""
        return cls._summarization_model_max_tokens

    @classmethod
    def get_summarization_model_name(cls) -> str:
        """Get the name of the summarization model."""
        return cls._summarization_model_name

    # ===== MODEL GETTERS =====
    @classmethod
    def get_device(cls) -> torch.device:
        """Detect GPU (CUDA) if available, else fallback to CPU."""
        if cls._device is None:
            try:
                # Use the centralized device detection
                from app.core.concurrency.device import get_torch_device

                cls._device = get_torch_device()
                cls._logger.info(f"model_manager.py:Using device: {cls._device}")
            except Exception as e:
                cls._logger.error(f"model_manager.py:Failed to get device: {e}")
                # Fallback to CPU if device detection fails
                cls._device = torch.device("cpu")
                cls._logger.warning(
                    f"model_manager.py:Falling back to CPU device due to error: {e}"
                )
        return cls._device

    # @classmethod
    # def _ensure_nltk_models(cls):
    #     """Ensure NLTK models are available for production with multi-language support."""
    #     try:
    #         import nltk
    #         from nltk.tokenize import sent_tokenize

    #         # Test core models
    #         test_text = "This is a test sentence. Another one."
    #         sent_tokenize(test_text, language="english")

    #         # Test additional language models
    #         test_languages = ["german", "french", "spanish", "italian"]
    #         for lang in test_languages:
    #             try:
    #                 sent_tokenize("Test sentence.", language=lang)
    #                 cls._logger.info(f"NLTK model for {lang} is available")
    #             except LookupError:
    #                 cls._logger.warning(
    #                     f"NLTK model for {lang} not found, will use fallback tokenization"
    #                 )

    #         cls._logger.info("NLTK models validation completed")

    #     except LookupError:
    #         cls._logger.warning("NLTK models not found, downloading...")
    #         try:
    #             nltk.download("punkt", quiet=True)
    #             nltk.download("punkt_tab", quiet=True)
                
    #             cls._logger.info("NLTK models downloaded successfully")
    #         except Exception as e:
    #             cls._logger.error(f"Failed to download NLTK models: {e}")
    #             raise RuntimeError("NLTK models required for production operation")

    #     except Exception as e:
    #         cls._logger.error(f"Unexpected error checking NLTK models: {e}")
    #         raise

    # @classmethod
    # def get_summarization_model(cls):
    #     """Return BART summarization model (lazy init with CPU/GPU)."""
    #     if cls._summarization_model is None or cls._summarization_tokenizer is None:
    #         cls._ensure_nltk_models()
    #         cls.__load_summarization_model()

    #     # Validate that all components are properly loaded
    #     if cls._summarization_model is None:
    #         raise RuntimeError("Summarization model failed to load")
    #     if cls._summarization_tokenizer is None:
    #         raise RuntimeError("Summarization tokenizer failed to load")
    #     if cls._device is None:
    #         raise RuntimeError("Device configuration failed to load")

    #     cls._logger.debug(
    #         f"model_manager.py:Returning summarization model components: model={type(cls._summarization_model)}, tokenizer={type(cls._summarization_tokenizer)}, device={cls._device}"
    #     )
    #     return cls._summarization_model, cls._summarization_tokenizer, cls._device

    @classmethod
    def get_embedding_model(cls) -> SentenceTransformer:
        """Return embedding model (thread-safe singleton)."""
        if cls._embedding_model is None:
            with cls._embedding_lock:
                if cls._embedding_model is None:
                    cls.__load_embedding_model()
        return cls._embedding_model


    # ===== INTERNAL LOADERS =====
    @classmethod
    def __load_embedding_model(cls):
        """Load embedding model onto GPU if available, else CPU (thread-safe)."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._logger.info(
                f"model_manager.py:Loading embedding model '{cls._embedding_model_name}' on {device}"
            )
            cls._embedding_model = SentenceTransformer(
                cls._embedding_model_name,
                cache_folder=cls.get_model_cache_dir(),
                device=device,
            )
            cls._logger.info(f"model_manager.py:Embedding model loaded on {device}")
        except Exception as e:
            cls._logger.error(f"model_manager.py:Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")


