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
This module provides utilities to normalize and translate multilingual text to English.
"""

import re
import time
import unicodedata

from app.config import get_service_logger, get_settings
from app.core.exceptions import TranslationException
from app.constants import ISO_TO_NLLB_MERGED
from app.singleton import ModelManager, NLLBTranslatorSingleton


class Translator:
    """
    Provides utilities to normalize and translate multilingual text to English.
    """

    def __init__(self, max_chars=2000, retries=3, delay=2):
        self.max_chars = max_chars
        self.retries = retries
        self.delay = delay
        self.nllb_translator = NLLBTranslatorSingleton()
        self.translator = ModelManager.get_translator(lang="en")
        self.logger = get_service_logger("Translator")
        self.settings = get_settings()

    def ensure_english(self, text: str) -> str:
        """Return text as-is if it's English, otherwise return translated-to-English version."""
        lang = ModelManager.detect_language(text[:1000])
        if lang == "en":
            return text

        if self.settings.debug:
            return self.translate(text, lang, "en")
        else:
            return self.translate_prod(text, lang, "en")

    def translate_prod(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Try NLLB first, then fallback to Google Translate if NLLB fails or is unavailable.
        """
        try:
            # NLLB First
            try:
                if (
                    source_lang in ISO_TO_NLLB_MERGED
                    and target_lang in ISO_TO_NLLB_MERGED
                ):
                    self.logger.info(
                        f"[Translation] Using NLLB: {source_lang} → {target_lang}"
                    )
                    src_nllb = ISO_TO_NLLB_MERGED[source_lang]
                    tgt_nllb = ISO_TO_NLLB_MERGED[target_lang]
                    return self.nllb_translate(text, src_nllb, tgt_nllb)
                else:
                    self.logger.warning(
                        f"[TranslationWarning] NLLB not available for {source_lang} → {target_lang}. Trying Google Translate."
                    )
            except Exception as nllb_error:
                self.logger.warning(
                    f"[TranslationWarning] NLLB failed: {nllb_error}. Falling back to Google Translate."
                )

            # Google Fallback
            self.logger.info(
                f"[Translation] Using Google Translate: {source_lang} → {target_lang}"
            )
            return (
                self.google_translate_to_english(text)
                if target_lang == "en"
                else self.google_translate(text, source_lang, target_lang)
            )

        except Exception as e:
            self.logger.error(
                f"[TranslationError] Could not translate text from '{source_lang}' to '{target_lang}'"
            )
            self.logger.error(f"Translation failed: {e}", exc_info=True)
            raise TranslationException(f"Batch translation failed: {str(e)}")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Try Google Translate first, then fallback to NLLB if Google fails or is unavailable.
        """
        try:

            # Google First
            try:
                self.logger.info(
                    f"[Translation] Using Google Translate: {source_lang} → {target_lang}"
                )
                result = self.google_translate_to_english(text)
                return result
            except Exception as google_error:
                self.logger.warning(
                    f"[TranslationWarning] Google failed: {google_error}. Falling back to NLLB."
                )

            # NLLB Fallback
            hf_model_used = False
            if source_lang in ISO_TO_NLLB_MERGED and target_lang in ISO_TO_NLLB_MERGED:
                self.logger.info(
                    f"[Translation] Using NLLB: {source_lang} → {target_lang}"
                )
                src_nllb = ISO_TO_NLLB_MERGED[source_lang]
                tgt_nllb = ISO_TO_NLLB_MERGED[target_lang]
                hf_model_used = True
                return self.nllb_translate(text, src_nllb, tgt_nllb)

            # If neither works, return original text
            self.logger.warning(
                f"[TranslationWarning] No valid translation method available for {source_lang} → {target_lang}. Returning original text."
            )
            return text

        except Exception as e:
            self.logger.error(
                f"[TranslationError] Could not translate text from '{source_lang}' to '{target_lang}'"
            )
            self.logger.error(f"Translation failed: {e}", exc_info=True)
            raise TranslationException(f"Batch translation failed: {str(e)}")

    def nllb_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate the text to English.
        """
        try:
            # split the text into chunks, as the google translator has a limit of 2000 characters?
            self.logger.info(f"[Translation] Using NLLB: {source_lang} → {target_lang}")
            chunks = self.__split_text_smart(text, 400)
            translated_chunks = [
                self.nllb_translator.translate(chunk, source_lang, target_lang)
                for chunk in chunks
            ]
            return "\n\n".join(translated_chunks)
        except Exception as e:
            self.logger.error(f"[CRITICAL] Full translation failed: {e}")
            return text

    def google_translate_to_english(self, text: str) -> str:
        """
        Translate the text to English.
        """
        try:
            # split the text into chunks, as the google translator has a limit of 2000 characters?
            self.logger.info(f"[Translation] Using Google Translate")
            chunks = self.__split_text_smart(text, self.max_chars)
            translated_chunks = [
                self.__safe_translate(self.__clean_text(chunk)) for chunk in chunks
            ]
            return "\n\n".join(translated_chunks)
        except Exception as e:
            self.logger.error(f"[CRITICAL] Full translation failed: {e}")
            return text

    def __split_text_smart(self, text: str, max_chars: int = 2000) -> list:
        """
        Split the text into chunks, as the google translator has a limit of 2000 characters?
        """
        self.logger.info(
            f"[Translation] Splitting text into chunks: {max_chars} characters per chunk"
        )
        sentence_endings = re.split(r"(?<=[\.\!\?।؟。！？])\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentence_endings:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(sentence) > max_chars:
                    for i in range(0, len(sentence), max_chars):
                        chunks.append(sentence[i : i + max_chars])
                    current_chunk = ""
                else:
                    current_chunk = sentence + " "
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def __clean_text(self, chunk: str) -> str:
        return unicodedata.normalize("NFKC", chunk).strip()

    def __safe_translate(self, chunk: str) -> str:
        for attempt in range(self.retries):
            try:
                return self.translator.translate(chunk)
            except Exception as e:
                print(f"[Attempt {attempt + 1}/{self.retries}] Translation failed: {e}")
                time.sleep(self.delay)
        print(f"[Fallback] Using original chunk:\n{chunk[:80]}...")
        return chunk
