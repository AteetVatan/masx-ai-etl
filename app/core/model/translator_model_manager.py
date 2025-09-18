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
Translation model manager with pooling capabilities.

Manages NLLB translation models with controlled concurrency and memory management.
"""

import os
import nltk
import torch
from typing import Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

from .abstract_model import AbstractModel, ModelInstance


class TranslatorModelManager(AbstractModel[AutoModelForSeq2SeqLM]):
    """
    Manager for NLLB translation models with pooling capabilities.

    Provides controlled access to translation model instances with automatic
    memory management and GPU/CPU optimization.
    """

    def __init__(self, settings: Optional[Any] = None):
        super().__init__(settings)
        self._model_name = "facebook/nllb-200-distilled-600M"
        self._max_tokens = 512
        self._gen_config: Optional[GenerationConfig] = None

    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        return self._model_name

    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "translation"

    @property
    def max_tokens(self) -> int:
        """Return the maximum number of tokens for translation."""
        return self._max_tokens

    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens for translation."""
        return self._max_tokens

    def get_tokenizer(self) -> AutoTokenizer:
        """Return the tokenizer for translation."""
        return self._load_tokenizer()

    def get_device(self) -> torch.device:
        """Return the device for translation."""
        return super().get_device()

    def get_model(self) -> AutoModelForSeq2SeqLM:
        """Return the model for translation."""
        return self._load_model()

    def _get_model_vram_estimate(self) -> int:
        """Get estimated VRAM usage for NLLB-200-distilled-600M model."""
        return 1.5 * 1024**3  # 1.5GB for NLLB-200-distilled-600M

    def _ensure_nltk_models(self) -> None:
        """Ensure NLTK models are available for production with multi-language support."""
        try:
            # Test core models
            test_text = "This is a test sentence. Another one."
            nltk.sent_tokenize(test_text)

            # Test additional language models
            test_languages = ["german", "french", "spanish", "italian"]
            for lang in test_languages:
                try:
                    nltk.sent_tokenize("Test sentence.", language=lang)
                    self.logger.info(f"NLTK model for {lang} is available")
                except LookupError:
                    self.logger.warning(
                        f"NLTK model for {lang} not found, will use fallback tokenization"
                    )

            self.logger.info("NLTK models validation completed")

        except LookupError:
            self.logger.warning("NLTK models not found, downloading...")
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                self.logger.info("NLTK models downloaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to download NLTK models: {e}")
                raise RuntimeError("NLTK models required for production operation")
        except Exception as e:
            self.logger.error(f"Unexpected error checking NLTK models: {e}")
            raise

    def _load_model(self) -> AutoModelForSeq2SeqLM:
        """Load and return a NLLB translation model instance.
        https://huggingface.co/facebook/nllb-200-distilled-600M
        """
        try:

            device = self.get_device()
            use_fp16 = device.type == "cuda"

            try:
                # Load model with appropriate precision and memory optimization
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self._model_name,
                    cache_dir=self._get_model_cache_dir(),
                    torch_dtype=torch.float16 if use_fp16 else torch.float32,
                    low_cpu_mem_usage=False if self.settings.is_production else True,
                    local_files_only=True,
                )
            except RuntimeError as load_err:
                self.logger.warning(f"FP16 load failed, retrying with FP32: {load_err}")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self._model_name,
                    cache_dir=self._get_model_cache_dir(),
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False if self.settings.is_production else True,
                    local_files_only=True,
                )

            # Move to device and configure
            model = model.to(device)
            model.config.use_cache = True
            model.eval()

            if use_fp16:
                torch.set_float32_matmul_precision("high")

            self.logger.info(
                f"Loaded translation model '{self._model_name}' on {device}"
            )
            return model

        except Exception as e:
            self.logger.error(f"Failed to load translation model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load translation model: {e}")

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load and return a NLLB tokenizer instance.
        https://huggingface.co/facebook/nllb-200-distilled-600M
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                cache_dir=self._get_model_cache_dir(),
                use_fast=True,
                local_files_only=True,
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
        """Get the generation configuration for translation."""
        if self._gen_config is None:
            self._gen_config = GenerationConfig(
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                do_sample=False,
                max_length=512,
                min_length=1,
            )
        return self._gen_config

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        instance: Optional[ModelInstance[AutoModelForSeq2SeqLM]] = None,
    ) -> str:
        """
        Translate text using a model instance.

        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'eng_Latn')
            target_lang: Target language code (e.g., 'spa_Latn')
            instance: Model instance to use (if None, will acquire one)

        Returns:
            Translated text
        """
        if instance is None:
            with self.acquire() as instance:
                return self._perform_translation(
                    text, source_lang, target_lang, instance
                )
        else:
            return self._perform_translation(text, source_lang, target_lang, instance)

    def _perform_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        instance: ModelInstance[AutoModelForSeq2SeqLM],
    ) -> str:
        """Perform the actual translation using the model instance."""
        try:
            tokenizer = instance.tokenizer
            model = instance.model
            device = instance.device

            # Set source and target languages in tokenizer
            tokenizer.src_lang = source_lang

            # Tokenize input
            inputs = tokenizer(
                text,
                max_length=self._max_tokens,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=self.get_generation_config(),
                    forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
                )

            # Decode output
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation.strip()

        except Exception as e:
            self.logger.error(f"Failed to translate text: {e}")
            raise RuntimeError(f"Translation failed: {e}")

    def translate_to_english(
        self,
        text: str,
        source_lang: str,
        instance: Optional[ModelInstance[AutoModelForSeq2SeqLM]] = None,
    ) -> str:
        """
        Convenience method to translate text to English.

        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'spa_Latn', 'fra_Latn')
            instance: Model instance to use (if None, will acquire one)

        Returns:
            Translated text in English
        """
        return self.translate_text(text, source_lang, "eng_Latn", instance)

    def translate_from_english(
        self,
        text: str,
        target_lang: str,
        instance: Optional[ModelInstance[AutoModelForSeq2SeqLM]] = None,
    ) -> str:
        """
        Convenience method to translate text from English.

        Args:
            text: English text to translate
            target_lang: Target language code (e.g., 'spa_Latn', 'fra_Latn')
            instance: Model instance to use (if None, will acquire one)

        Returns:
            Translated text in target language
        """
        return self.translate_text(text, "eng_Latn", target_lang, instance)

    def get_supported_languages(self) -> dict:
        """
        Get the list of supported languages for NLLB model.

        Returns:
            Dictionary mapping language codes to language names
        """
        # Common language codes supported by NLLB-200
        return {
            "eng_Latn": "English",
            "spa_Latn": "Spanish",
            "fra_Latn": "French",
            "deu_Latn": "German",
            "ita_Latn": "Italian",
            "por_Latn": "Portuguese",
            "rus_Cyrl": "Russian",
            "zho_Hans": "Chinese (Simplified)",
            "zho_Hant": "Chinese (Traditional)",
            "jpn_Jpan": "Japanese",
            "kor_Hang": "Korean",
            "ara_Arab": "Arabic",
            "hin_Deva": "Hindi",
            "ben_Beng": "Bengali",
            "urd_Arab": "Urdu",
            "nld_Latn": "Dutch",
            "pol_Latn": "Polish",
            "tur_Latn": "Turkish",
            "swe_Latn": "Swedish",
            "nor_Latn": "Norwegian",
            "dan_Latn": "Danish",
            "fin_Latn": "Finnish",
            "ell_Grek": "Greek",
            "heb_Hebr": "Hebrew",
            "tha_Thai": "Thai",
            "vie_Latn": "Vietnamese",
            "ind_Latn": "Indonesian",
            "msa_Latn": "Malay",
            "tgl_Latn": "Filipino",
            "ukr_Cyrl": "Ukrainian",
            "ces_Latn": "Czech",
            "hun_Latn": "Hungarian",
            "ron_Latn": "Romanian",
            "bul_Cyrl": "Bulgarian",
            "hrv_Latn": "Croatian",
            "slv_Latn": "Slovenian",
            "slk_Latn": "Slovak",
            "est_Latn": "Estonian",
            "lav_Latn": "Latvian",
            "lit_Latn": "Lithuanian",
        }
