"""
This module contains the Model class, which is a singleton class that loads and manages the models.
"""

import os
import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from lingua import Language, LanguageDetectorBuilder
import langid
import fasttext
import requests
from config import get_service_logger


# local directory path where Hugging Face will store the downloaded model weights, tokenizers, and configuration files — instead of downloading them every time.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE = os.path.join(BASE_DIR, "..", ".hf_cache")  # or any central path


class ModelManager:
    """Singleton class to load and manage models and translators."""

    _summarization_model: AutoModelForSeq2SeqLM | None = None
    _summarization_tokenizer: AutoTokenizer | None = None
    _translator: GoogleTranslator | None = None
    _fasttext_lang_detector: fasttext.FastText._FastText | None = None
    _device: torch.device | None = None
    _summarization_model_max_tokens: int = 1024
    # transformer model for text summarization
    _summarization_model_name: str = "facebook/bart-large-cnn"
    
    # transformer model for text embedding
    _embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    _embedding_model: SentenceTransformer | None = None
    _embedding_lock = threading.Lock()
    _logger = get_service_logger("ModelManager")
    
    # path helpers
    @classmethod
    def get_base_dir(cls) -> str:
        """Root base directory of this module"""
        return os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def get_model_cache_dir(cls) -> str:
        """Shared cache folder for Hugging Face models"""
        return os.path.join(cls.get_base_dir(), "..", ".hf_cache")

    @classmethod
    def get_fasttext_model_path(cls) -> str:
        """Location of the FastText language ID model (lid.176.bin)
        pre-trained by Facebook AI and supports language detection for 176 languages.
        """
        return os.path.join(cls.get_base_dir(), "..", "lid.176.bin")

    @classmethod
    def get_summarization_model_max_tokens(cls) -> int:
        """Get the maximum number of tokens for the summarization model."""
        return cls._summarization_model_max_tokens

    @classmethod
    def get_summarization_model_name(cls) -> str:
        """Get the name of the summarization model."""
        return cls._summarization_model_name

    @classmethod
    def get_summarization_model(
        cls,
    ) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer, torch.device]:
        """
        Get the BART model, tokenizer, and device.
        Lazily initializes if not already loaded.
        """
        if cls._summarization_model is None or cls._summarization_tokenizer is None:
            cls.__load_summarization_model()
        return cls._summarization_model, cls._summarization_tokenizer, cls._device
    
    @classmethod
    def get_embedding_model(cls) -> SentenceTransformer:
        """
        Get the SentenceTransformer embedding model.
        Lazily initializes if not already loaded (thread-safe).
        """
        if cls._embedding_model is None:
            with cls._embedding_lock:  # Lock ensures only one thread initializes the model
                if cls._embedding_model is None:  # Double-check inside lock
                    cls.__load_embedding_model()
        return cls._embedding_model

    @classmethod
    def get_translator(cls, lang="en") -> GoogleTranslator:
        """
        Get the GoogleTranslator instance.
        Lazily initializes if not already loaded.
        """
        if cls._translator is None:
            cls.__load_translator(lang)
        return cls._translator

    @classmethod
    def get_lang_detector_fasttext(cls) -> fasttext.FastText._FastText:
        """Get the FastText language ID model."""
        if cls._fasttext_lang_detector is None:
            cls.__load_fasttext_lang_detector()
        return cls._fasttext_lang_detector

    @classmethod
    def detect_lang_fasttext(cls, text: str) -> str:
        """Detect language using fasttext model (returns ISO 639-1 code like 'en', 'fr')."""
        try:
            detector = cls.get_lang_detector_fasttext()
            prediction = detector.predict(text.strip().replace("\n", " "))[0][0]
            return prediction.replace("__label__", "")
        except Exception as e:
            raise RuntimeError(f"FastText language detection failed: {e}")
        
    
    @classmethod
    def get_lingua_detector(cls, languages=None):
        builder = (
            LanguageDetectorBuilder
            .from_all_languages()  # or .from_languages(...subset...) if you want
        )
        return builder.build()

    @classmethod
    def get_langid_identifier(cls, langs=None, norm_probs=True):
        identifier = langid.langid.LanguageIdentifier.from_modelstring(
            langid.langid.model, norm_probs=norm_probs
        )
        if langs:
            identifier.set_languages(langs)
        return identifier


    @classmethod
    def detect_lang_lingua(cls, text: str):
        detector = cls.get_lingua_detector()
        lang = detector.detect_language_of(text)
        return lang.iso_code_639_1.name if lang else None

    @classmethod
    def detect_lang_langid(cls, text: str, langs=None):
        identifier = cls.get_langid_identifier(langs=langs)
        lang, prob = identifier.classify(text)
        return lang, prob
    
    @classmethod
    def detect_language(cls, text: str) -> dict:
        try:
            lang, li_conf = cls.detect_lang_langid(text)
            if li_conf < 0.99:
                lang = cls.detect_lang_lingua(text)
            return lang.lower()
        except Exception as e:           
            lang = cls.detect_lang_fasttext(text)
            return lang.lower()        
       

    # ========== Internal Loaders ==========

    @classmethod
    def __load_summarization_model(cls):
        """Load the BART model for summarization onto GPU if available."""
        try:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            cls._summarization_tokenizer = AutoTokenizer.from_pretrained(
                cls._summarization_model_name, cache_dir=cls.get_model_cache_dir()
            )

            cls._summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
                cls._summarization_model_name,
                cache_dir=cls.get_model_cache_dir(),
                torch_dtype=(
                    torch.float16 if cls._device.type == "cuda" else torch.float32
                ),
            ).to(cls._device)

        except Exception as e:
            cls._logger.error(f"Failed to load BART model: {e}")
            raise RuntimeError(f"Failed to load BART model: {e}")
        
    @classmethod
    def __load_embedding_model(cls):
        """Load the 'all-mpnet-base-v2' model from SentenceTransformers."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._embedding_model = SentenceTransformer(cls._embedding_model_name, cache_folder=cls.get_model_cache_dir(), device=device)
        except Exception as e:
            cls._logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")
        
    @classmethod
    def __load_embedding_model_1(cls):
        """Manually load embedding model to avoid meta tensor issue."""
        import torch
        from transformers import AutoModel, AutoTokenizer
        from sentence_transformers import SentenceTransformer, models

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._logger.info(f"Manually loading embedding model '{cls._embedding_model_name}' on {device}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                cls._embedding_model_name,
                cache_dir=cls.get_model_cache_dir()
            )

            # Load encoder (transformer backbone) safely on device
            transformer = AutoModel.from_pretrained(
                cls._embedding_model_name,
                cache_dir=cls.get_model_cache_dir()
            ).to(device)

            # Wrap encoder in SentenceTransformer's modules
            transformer_model = models.Transformer(transformer=transformer, tokenizer=tokenizer)
            pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension())

            # Build SentenceTransformer pipeline manually
            cls._embedding_model = SentenceTransformer(modules=[transformer_model, pooling_model], device=device)

            cls._logger.info(f"Embedding model loaded successfully on {device}")

        except Exception as e:
            cls._logger.error(f"Failed to load embedding model manually: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")


    @classmethod
    def __load_translator(cls, lang):
        """Load the GoogleTranslator for multilingual-to-target language translation."""
        try:
            cls._translator = GoogleTranslator(source="auto", target=lang)
        except Exception as e:
            cls._logger.error(f"Failed to load GoogleTranslator: {e}")
            raise RuntimeError(f"Failed to load GoogleTranslator: {e}")

    @classmethod
    def __load_fasttext_lang_detector(cls):
        try:
            fasttext_path = cls.get_fasttext_model_path()
            if not os.path.exists(fasttext_path):
                cls.__download_fasttext_model(fasttext_path)
            cls._fasttext_lang_detector = fasttext.load_model(fasttext_path)
        except Exception as e:
            cls._logger.error(f"Failed to load fasttext language model: {e}")
            raise RuntimeError(f"Failed to load fasttext language model: {e}")

    @classmethod
    def __download_fasttext_model(cls, save_path: str):
        """Download FastText lid.176.bin model if not already present."""
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

        try:
            print(f"Downloading FastText model to: {save_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            cls._logger.error(f"Failed to download FastText model: {e}")
            raise RuntimeError(f"Failed to download FastText model: {e}")
