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

import torch
from threading import Lock
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from app.config import get_service_logger, get_settings
from app.core.concurrency import InferenceRuntime, RuntimeConfig


class NLLBTranslatorSingleton:
    """
    Singleton class for loading and serving the NLLB-200 multilingual translation model.
    Supports efficient reuse across threads, agents, or pipelines using InferenceRuntime.
    """

    _instance = None
    _lock: Lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NLLBTranslatorSingleton, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model_name = "facebook/nllb-200-distilled-600M"
        self.settings = get_settings()
        self.logger = get_service_logger("NLLBTranslatorSingleton")

        # Initialize inference runtime for translation
        self.inference_runtime: Optional[InferenceRuntime] = None

        # Cache translation pipelines to avoid re-init
        self.pipelines: Dict[str, pipeline] = {}

        self._initialized = True

    async def _initialize_inference_runtime(self):
        """Initialize the inference runtime for translation."""
        try:
            # Create runtime config optimized for translation
            config = RuntimeConfig(
                gpu_batch_size=self.settings.gpu_batch_size,
                gpu_max_delay_ms=self.settings.gpu_max_delay_ms,
                gpu_queue_size=self.settings.gpu_queue_size,
                gpu_timeout=self.settings.gpu_timeout,
                gpu_use_fp16=self.settings.gpu_use_fp16,
                gpu_enable_warmup=self.settings.gpu_enable_warmup,
                cpu_max_threads=self.settings.cpu_max_threads,
                cpu_max_processes=self.settings.cpu_max_processes,
            )

            # Create and start inference runtime
            self.inference_runtime = InferenceRuntime(
                model_loader=self._get_translation_model_loader, config=config
            )

            await self.inference_runtime.start()
            self.logger.info("nllb_translator_singleton.py:NLLBTranslator:Inference runtime initialized for NLLB translation")

        except Exception as e:
            self.logger.error(f"nllb_translator_singleton.py:NLLBTranslator:Failed to initialize inference runtime: {e}")
            raise

    def _get_translation_model_loader(self):
        """Model loader function for the inference runtime."""
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Return both tokenizer and model for the inference runtime
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
        Translate text from source to target language using InferenceRuntime.

        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code

        Returns:
            Translated text
        """
        try:
            # Initialize inference runtime if not already done
            if not self.inference_runtime:
                await self._initialize_inference_runtime()

            # Prepare payload for translation
            payload = {"text": text, "src_lang": src_lang, "tgt_lang": tgt_lang}

            # Use inference runtime for translation
            result = await self.inference_runtime.infer(payload)

            if isinstance(result, Exception):
                self.logger.error(f"nllb_translator_singleton.py:NLLBTranslator:Translation failed: {result}")
                # Fallback to synchronous translation
                return self._translate_sync(text, src_lang, tgt_lang)

            return result

        except Exception as e:
            self.logger.error(f"nllb_translator_singleton.py:NLLBTranslator:Translation failed: {e}")
            # Fallback to synchronous translation
            return self._translate_sync(text, src_lang, tgt_lang)

    def _translate_sync(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Synchronous fallback translation method.
        """
        try:
            pipeline = self._get_pipeline(src_lang, tgt_lang)
            result = pipeline(text)
            return result[0]["translation_text"]
        except Exception as e:
            self.logger.error(f"nllb_translator_singleton.py:NLLBTranslator:Synchronous translation failed: {e}")
            return text  # Return original text if translation fails

    async def translate_batch(
        self, texts: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        """
        Translate a batch of texts using InferenceRuntime with micro-batching.

        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code

        Returns:
            List of translated texts
        """
        try:
            # Initialize inference runtime if not already done
            if not self.inference_runtime:
                await self._initialize_inference_runtime()

            # Prepare payloads for batch processing
            payloads = [
                {"text": text, "src_lang": src_lang, "tgt_lang": tgt_lang, "index": i}
                for i, text in enumerate(texts)
            ]

            # Use inference runtime for batch translation
            results = await self.inference_runtime.infer_many(payloads)

            # Process results
            translated_texts = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"nllb_translator_singleton.py:NLLBTranslator:Translation failed for text {i}: {result}")
                    # Fallback to synchronous translation for this item
                    translated_text = self._translate_sync(texts[i], src_lang, tgt_lang)
                    translated_texts.append(translated_text)
                else:
                    translated_texts.append(result)

            return translated_texts

        except Exception as e:
            self.logger.error(f"nllb_translator_singleton.py:NLLBTranslator:Batch translation failed: {e}")
            # Fallback to synchronous batch translation
            return [self._translate_sync(text, src_lang, tgt_lang) for text in texts]

    def translate_sync(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Synchronous wrapper for backward compatibility.

        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code

        Returns:
            Translated text
        """
        import asyncio

        return asyncio.run(self.translate(text, src_lang, tgt_lang))

    async def stop(self):
        """Stop the inference runtime."""
        if self.inference_runtime:
            await self.inference_runtime.stop()
            self.logger.info("nllb_translator_singleton.py:NLLBTranslator:NLLB translator inference runtime stopped")
