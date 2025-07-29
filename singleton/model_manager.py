"""
This module contains the Model class, which is a singleton class that loads and manages the models.
"""

"""
This module contains the Model class, which is a singleton class that loads and manages the models.
"""

import os
import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from lingua import LanguageDetectorBuilder
import langid
from config import get_service_logger

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
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._logger.info(f"Using device: {cls._device}")
        return cls._device

    @classmethod
    def get_summarization_model(cls):
        """Return BART summarization model (lazy init with CPU/GPU)."""
        if cls._summarization_model is None or cls._summarization_tokenizer is None:
            cls.__load_summarization_model()
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

            cls._logger.info(f"Loaded summarization model on {cls._device}")

        except Exception as e:
            cls._logger.error(f"Failed to load summarization model: {e}")
            raise RuntimeError(f"Failed to load summarization model: {e}")

    @classmethod
    def __load_embedding_model(cls):
        """Load embedding model onto GPU if available, else CPU (thread-safe)."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._logger.info(
                f"Loading embedding model '{cls._embedding_model_name}' on {device}"
            )
            cls._embedding_model = SentenceTransformer(
                cls._embedding_model_name,
                cache_folder=cls.get_model_cache_dir(),
                device=device,
            )
            cls._logger.info(f"Embedding model loaded on {device}")
        except Exception as e:
            cls._logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")

    @classmethod
    def __load_translator(cls, lang):
        try:
            cls._translator = GoogleTranslator(source="auto", target=lang)
        except Exception as e:
            cls._logger.error(f"Failed to load GoogleTranslator: {e}")
            raise RuntimeError(f"Failed to load GoogleTranslator: {e}")
