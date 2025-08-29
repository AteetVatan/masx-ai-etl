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

import langid
import torch

from deep_translator import GoogleTranslator
from lingua import LanguageDetectorBuilder
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import nltk
from nltk.tokenize import sent_tokenize

from app.config import get_service_logger

# local directory path where Hugging Face will store the downloaded model weights, tokenizers, and configuration files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE = os.path.join(BASE_DIR, "..", ".hf_cache")  # or any central path


class ModelManager:
    """Singleton class to load and manage models and translators with CPU/GPU detection."""

    _summarization_model: AutoModelForSeq2SeqLM | None = None
    _summarization_tokenizer: AutoTokenizer | None = None
    _translator: GoogleTranslator | None = None
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

    @classmethod
    def _ensure_nltk_models(cls):
        """Ensure NLTK models are available for production with multi-language support."""
        try:
            import nltk
            from nltk.tokenize import sent_tokenize

            # Test core models
            test_text = "This is a test sentence. Another one."
            sent_tokenize(test_text, language="english")

            # Test additional language models
            test_languages = ["german", "french", "spanish", "italian"]
            for lang in test_languages:
                try:
                    sent_tokenize("Test sentence.", language=lang)
                    cls._logger.info(f"NLTK model for {lang} is available")
                except LookupError:
                    cls._logger.warning(
                        f"NLTK model for {lang} not found, will use fallback tokenization"
                    )

            cls._logger.info("NLTK models validation completed")

        except LookupError:
            cls._logger.warning("NLTK models not found, downloading...")
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("averaged_perceptron_tagger", quiet=True)
                cls._logger.info("NLTK models downloaded successfully")
            except Exception as e:
                cls._logger.error(f"Failed to download NLTK models: {e}")
                raise RuntimeError("NLTK models required for production operation")

        except Exception as e:
            cls._logger.error(f"Unexpected error checking NLTK models: {e}")
            raise

    @classmethod
    def get_summarization_model(cls):
        """Return BART summarization model (lazy init with CPU/GPU)."""
        if cls._summarization_model is None or cls._summarization_tokenizer is None:
            cls._ensure_nltk_models()
            cls.__load_summarization_model()

        # Validate that all components are properly loaded
        if cls._summarization_model is None:
            raise RuntimeError("Summarization model failed to load")
        if cls._summarization_tokenizer is None:
            raise RuntimeError("Summarization tokenizer failed to load")
        if cls._device is None:
            raise RuntimeError("Device configuration failed to load")

        cls._logger.debug(
            f"model_manager.py:Returning summarization model components: model={type(cls._summarization_model)}, tokenizer={type(cls._summarization_tokenizer)}, device={cls._device}"
        )
        return cls._summarization_model, cls._summarization_tokenizer, cls._device

    @classmethod
    def get_embedding_model(cls) -> SentenceTransformer:
        """Return embedding model (thread-safe singleton)."""
        if cls._embedding_model is None:
            with cls._embedding_lock:
                if cls._embedding_model is None:
                    cls.__load_embedding_model()
        return cls._embedding_model

    @classmethod
    def get_translator(cls, lang="en") -> GoogleTranslator:
        if cls._translator is None:
            cls.__load_translator(lang)
        return cls._translator

    # ===== LANGUAGE DETECTION =====
    @classmethod
    def detect_language(cls, text: str) -> str:
        """Detect language using langid, fallback to lingua or fasttext."""
        try:
            lang, confidence = cls.detect_lang_langid(text)
            if confidence < 0.99:
                lang = cls.detect_lang_lingua(text)
            return lang.lower()
        except Exception:
            return cls.detect_lang_fasttext(text).lower()

    @classmethod
    def get_lingua_detector(cls, languages=None):
        builder = (
            LanguageDetectorBuilder.from_all_languages()  # or .from_languages(...subset...) if you want
        )
        return builder.build()

    @classmethod
    def detect_lang_lingua(cls, text: str):
        lang = cls.get_lingua_detector().detect_language_of(text)
        return lang.iso_code_639_1.name if lang else None

    @classmethod
    def detect_lang_langid(cls, text: str, langs=None):
        identifier = cls.get_langid_identifier(langs=langs)
        lang, prob = identifier.classify(text)
        return lang, prob

    @classmethod
    def get_langid_identifier(cls, langs=None, norm_probs=True):
        identifier = langid.langid.LanguageIdentifier.from_modelstring(
            langid.langid.model, norm_probs=norm_probs
        )
        if langs:
            identifier.set_languages(langs)
        return identifier

    # ===== INTERNAL LOADERS =====

    @classmethod
    def __load_summarization_model(cls):
        """Load summarization model with safe precision, tokenizer alignment, and sane generation defaults."""
        try:
            device = cls.get_device()
            use_fp16 = device.type == "cuda"

            # 1) Tokenizer (fast) + pad token safety
            tok = AutoTokenizer.from_pretrained(
                cls._summarization_model_name,
                cache_dir=cls.get_model_cache_dir(),
                use_fast=True,
            )

            # Some BART checkpoints don’t define pad_token explicitly.
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token

            # 2) Model with dtype & memory‑efficient attention
            model = AutoModelForSeq2SeqLM.from_pretrained(
                cls._summarization_model_name,
                cache_dir=cls.get_model_cache_dir(),
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
                low_cpu_mem_usage=True,
            )

            # Enable SDPA where available (PyTorch 2+)
            try:
                model = model.to(device)
                model.config.use_cache = True
            except Exception:
                model = model.to(device)

            # 3) Tokenizer/model vocab alignment (prevents IndexError on new tokens)
            if len(tok) != model.get_input_embeddings().num_embeddings:
                model.resize_token_embeddings(len(tok))

            # 4) Derive safe source length from positional embeddings
            max_pos = int(getattr(model.config, "max_position_embeddings", 1024))
            # buffer for special tokens; keep it conservative
            cls._max_src_len = max(32, max_pos - 4)

            # 5) Generation defaults (modern API; avoids overflows)
            cls._gen_cfg = GenerationConfig(
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,
            )

            # 6) Finalize
            model.eval()
            if use_fp16:
                # prefer fp16 matmul; TF32 for Ampere+ gives speedups too
                torch.set_float32_matmul_precision("high")

            cls._summarization_tokenizer = tok
            cls._summarization_model = model
            cls._device = device

            cls._logger.info(
                f"model_manager.py:Loaded summarization model '{cls._summarization_model_name}' "
                f"on {device} (dtype={model.dtype}, max_src_len={cls._max_src_len})"
            )

        except Exception as e:
            cls._logger.error(
                f"model_manager.py:Failed to load summarization model: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to load summarization model: {e}")

    @classmethod
    def __load_summarization_model_1(cls):
        """Load summarization model onto GPU if available, else CPU."""
        try:
            cls._device = cls.get_device()

            cls._summarization_tokenizer = AutoTokenizer.from_pretrained(
                cls._summarization_model_name,
                cache_dir=cls.get_model_cache_dir(),
            )

            cls._summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
                cls._summarization_model_name,
                cache_dir=cls.get_model_cache_dir(),
                torch_dtype=(
                    torch.float16 if cls._device.type == "cuda" else torch.float32
                ),
            ).to(cls._device)

            cls._logger.info(
                f"model_manager.py:Loaded summarization model on {cls._device}"
            )

        except Exception as e:
            cls._logger.error(
                f"model_manager.py:Failed to load summarization model: {e}"
            )
            raise RuntimeError(f"Failed to load summarization model: {e}")

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

    @classmethod
    def __load_translator(cls, lang):
        try:
            cls._translator = GoogleTranslator(source="auto", target=lang)
        except Exception as e:
            cls._logger.error(f"model_manager.py:Failed to load GoogleTranslator: {e}")
            raise RuntimeError(f"Failed to load GoogleTranslator: {e}")
