"""
This module contains the Model class, which is a singleton class that loads and manages the models.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator


# local directory path where Hugging Face will store the downloaded model weights, tokenizers, and configuration files — instead of downloading them every time.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE = os.path.join(BASE_DIR, "..", ".hf_cache")  # or any central path


class ModelManager:
    """Singleton class to load and manage models."""

    # Lazy initialization
    _bart_model: AutoModelForSeq2SeqLM | None = None
    _bart_tokenizer: AutoTokenizer | None = None
    _translator: GoogleTranslator | None = None

    # models getters
    @classmethod
    def get_bart_model(cls) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """Get the BART model and tokenizer."""
        if cls._bart_model is None or cls._bart_tokenizer is None:
            cls.__load_bart_model()
        return cls._bart_model, cls._bart_tokenizer

    @classmethod
    def get_translator(cls) -> GoogleTranslator:
        """Get the GoogleTranslator for translation."""
        if cls._translator is None:
            cls.__load_translator()
        return cls._translator

    @classmethod
    def __load_bart_model(cls):
        """Load the BART model for summarization."""
        try:
            cls._bart_tokenizer = AutoTokenizer.from_pretrained(
                "facebook/bart-large-cnn", cache_dir=MODEL_CACHE
            )

            cls._bart_model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/bart-large-cnn",
                cache_dir=MODEL_CACHE,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
             raise RuntimeError(f"Failed to load BART model: {e}")

    @classmethod
    def __load_translator(cls):
        """Load the GoogleTranslator for translation."""
        try:
            cls._translator = GoogleTranslator(source="auto", target="en")
        except Exception as e:
            raise RuntimeError(f"Failed to load GoogleTranslator: {e}")
