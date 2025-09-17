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

from lingua import LanguageDetectorBuilder
import langid


class LanguageUtils:
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language using langid, fallback to lingua or fasttext."""
        try:
            lang, confidence = LanguageUtils.detect_lang_langid(text)
            if confidence < 0.99:
                lang = LanguageUtils.detect_lang_lingua(text)
            return lang.lower()
        except Exception:
            return LanguageUtils.detect_lang_lingua(text).lower()

    @staticmethod
    def get_lingua_detector(languages=None):
        builder = (
            LanguageDetectorBuilder.from_all_languages()  # or .from_languages(...subset...) if you want
        )
        return builder.build()

    @staticmethod
    def detect_lang_lingua(text: str):
        lang = LanguageUtils.get_lingua_detector().detect_language_of(text)
        return lang.iso_code_639_1.name if lang else None

    @staticmethod
    def detect_lang_langid(text: str, langs=None):
        identifier = LanguageUtils.get_langid_identifier(langs=langs)
        lang, prob = identifier.classify(text)
        return lang, prob

    @staticmethod
    def get_langid_identifier(langs=None, norm_probs=True):
        identifier = langid.langid.LanguageIdentifier.from_modelstring(
            langid.langid.model, norm_probs=norm_probs
        )
        if langs:
            identifier.set_languages(langs)
        return identifier
