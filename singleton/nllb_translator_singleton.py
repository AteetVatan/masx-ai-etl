# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# │  Project: MASX AI – Strategic Agentic AI System              │
# │  All rights reserved.                                        │
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

# file: nllb_translator.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from threading import Lock
from typing import Dict
import torch


class NLLBTranslatorSingleton:
    """
    Singleton class for loading and serving the NLLB-200 multilingual translation model.
    Supports efficient reuse across threads, agents, or pipelines.
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

        # Detect device (GPU if available, else CPU)
        self.device = 0 if torch.cuda.is_available() else -1

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._initialized = True

        # Cache translation pipelines to avoid re-init
        self.pipelines: Dict[str, pipeline] = {}

    def _get_pipeline(self, src_lang: str, tgt_lang: str):
        """
        Returns a cached pipeline for the given src→tgt translation.
        """
        key = f"{src_lang}->{tgt_lang}"
        if key not in self.pipelines:
            self.pipelines[key] = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_length=512,
                device=self.device,
            )
        return self.pipelines[key]

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text from source to target language.
        """
        pipe = self._get_pipeline(src_lang, tgt_lang)
        result = pipe(text)
        return result[0]["translation_text"]
