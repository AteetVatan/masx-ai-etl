"""
This module contains the Model class, which is a singleton class that loads and manages the models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator


# local directory path where Hugging Face will store the downloaded model weights, tokenizers, and configuration files — instead of downloading them every time.
MODEL_CACHE = "./.hf_cache"


class Model:
    """Singleton class to load and manage models."""

    _bart = None
    _bart_tokenizer = None

    _translator = None

    # models getters

    @classmethod
    def get_bart_model(cls) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """Get the BART model and tokenizer."""
        return cls._bart, cls._bart_tokenizer

    @classmethod
    def get_translator(cls) -> GoogleTranslator:
        """Get the GoogleTranslator for translation."""
        return cls._translator

    @classmethod
    def load_models(cls):
        """Load all models."""
        cls.__load_bart()
        cls.__load_translator()

    @classmethod
    def __load_bart(cls):
        """Load the BART model for summarization."""
        if cls._bart and cls._bart_tokenizer:
            return

        cls._bart_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/bart-large-cnn", cache_dir=MODEL_CACHE
        )

        cls._bart = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn",
            cache_dir=MODEL_CACHE,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def __load_translator(cls):
        """Load the GoogleTranslator for translation."""
        if cls._translator:
            return

        cls._translator = GoogleTranslator(source="auto", target="en")
