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
This module contains the NLLBTranslatorSingleton class, which is a singleton class that loads and manages the NLLB-200 multilingual translation model.
"""

import logging
from threading import Lock
from typing import Dict, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from app.config import get_settings, get_service_logger


class NLLBTranslatorSingleton_old:
    """
    Singleton class for loading and serving the NLLB-200 multilingual translation model.
    Supports efficient reuse across threads, agents, or pipelines using InferenceRuntime.
    """

    _instance = None
    _lock: Lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NLLBTranslatorSingleton_old, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model_name = "facebook/nllb-200-distilled-600M"
        self.settings = get_settings()
        self.logger = get_service_logger("NLLBTranslatorSingleton")

        # Cache translation pipelines to avoid re-init
        self.pipelines: Dict[str, pipeline] = {}

        self._initialized = True

    def _get_translation_model_loader(self):
        """Model loader function for creating translation pipelines."""
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Return both tokenizer and model for pipeline creation
        return {"tokenizer": tokenizer, "model": model}

    def _get_pipeline(self, src_lang: str, tgt_lang: str):
        """
        Returns a cached pipeline for the given src→tgt translation.
        """
        key = f"{src_lang}->{tgt_lang}"
        if key not in self.pipelines:
            # Use the centralized device detection
            from app.core.concurrency.device import get_torch_device

            device = get_torch_device()

            # Load model and tokenizer
            model_data = self._get_translation_model_loader()
            tokenizer = model_data["tokenizer"]
            model = model_data["model"].to(device)

            self.pipelines[key] = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_length=512,
                device=device,
            )
        return self.pipelines[key]

    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text from source to target language using direct pipeline.

        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code

        Returns:
            Translated text
        """
        try:
            # Use direct pipeline translation instead of InferenceRuntime
            # to avoid infinite loop with GPU worker
            pipeline = self._get_pipeline(src_lang, tgt_lang)
            result = pipeline(text)
            return result[0]["translation_text"]

        except Exception as e:
            self.logger.error(
                f"nllb_translator_singleton.py:NLLBTranslator:Translation failed: {e}"
            )
            return text  # Return original text if translation fails

    def _translate_sync(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Synchronous translation method (now the primary method).
        """
        try:
            pipeline = self._get_pipeline(src_lang, tgt_lang)
            result = pipeline(text)
            return result[0]["translation_text"]
        except Exception as e:
            self.logger.error(
                f"nllb_translator_singleton.py:NLLBTranslator:Synchronous translation failed: {e}"
            )
            return text  # Return original text if translation fails

    async def translate_batch(
        self, texts: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        """
        Translate a batch of texts using direct pipeline processing.

        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code

        Returns:
            List of translated texts
        """
        try:
            # Get pipeline once for batch processing
            pipeline = self._get_pipeline(src_lang, tgt_lang)

            # Process texts in batches to avoid memory issues
            batch_size = 8  # Process 8 texts at a time
            translated_texts = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                try:
                    # Process batch
                    results = pipeline(batch)
                    # Extract translation text from results
                    batch_translations = [
                        result["translation_text"] for result in results
                    ]
                    translated_texts.extend(batch_translations)
                except Exception as e:
                    self.logger.error(
                        f"nllb_translator_singleton.py:NLLBTranslator:Batch translation failed for batch {i//batch_size}: {e}"
                    )
                    # Fallback to individual translation for failed batch
                    for text in batch:
                        try:
                            result = pipeline(text)
                            translated_texts.append(result[0]["translation_text"])
                        except Exception as e2:
                            self.logger.error(
                                f"nllb_translator_singleton.py:NLLBTranslator:Individual translation failed: {e2}"
                            )
                            translated_texts.append(
                                text
                            )  # Use original text as fallback

            return translated_texts

        except Exception as e:
            self.logger.error(
                f"nllb_translator_singleton.py:NLLBTranslator:Batch translation failed: {e}"
            )
            # Fallback to individual translation
            return [self._translate_sync(text, src_lang, tgt_lang) for text in texts]

    async def stop(self):
        """Stop the inference runtime."""
        # The inference_runtime object was removed, so this method is no longer needed.
        # Keeping it for now as it might be called externally, but it will do nothing.
        self.logger.warning(
            "nllb_translator_singleton.py:NLLBTranslator:Inference runtime object removed, stop method is no longer functional."
        )
