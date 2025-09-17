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
Summarization model manager with pooling capabilities.

Manages BART summarization models with controlled concurrency and memory management.
"""

import os
import nltk
import torch
from typing import Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

from .abstract_model import AbstractModel, ModelInstance


class SummarizationFinalizerModelManager(AbstractModel[AutoModelForSeq2SeqLM]):
    """
    Manager for FLAN-T5-Base finalization (clean + summarize) with pooling.
    Optimized for high-quality cleanup of noisy news text on CPU/GPU.
    """

    def __init__(self, settings: Optional[Any] = None):
        super().__init__(settings)
        self._model_name = "google/flan-t5-large"
        # T5 context ~512 tokens; keep headroom for instruction prefix
        self._max_tokens = 512
        self._gen_config: Optional[GenerationConfig] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_type(self) -> str:
        return "summarization"

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def get_max_tokens(self) -> int:
        return self._max_tokens

    def get_tokenizer(self) -> AutoTokenizer:
        return self._load_tokenizer()

    def get_device(self) -> torch.device:
        return super().get_device()

    def get_model(self) -> AutoModelForSeq2SeqLM:
        return self._load_model()

    def _get_model_vram_estimate(self) -> int:
        """
        Rough VRAM for flan-t5-base (~250M params).
        ~500MB params (fp16) + activations → budget ~1GB.
        """
        return 1 * 1024**3  # ~1GB

    
    def _preferred_dtype(self) -> torch.dtype:
        """
        Prefer BF16 on Ampere+ if available, else FP16 on CUDA, else FP32.
        """
        device = self.get_device()
        if device.type == "cuda":
            # BF16 support check (Ampere or newer)
            try:
                major, _ = torch.cuda.get_device_capability(device)
                if major >= 8:
                    return torch.bfloat16
            except Exception:
                pass
            return torch.float16
        return torch.float32

    def _load_model(self) -> AutoModelForSeq2SeqLM:
        """Load FLAN-T5-Base with correct dtype & device."""
        try:
            device = self.get_device()
            dtype = self._preferred_dtype()

            model = AutoModelForSeq2SeqLM.from_pretrained(
                self._model_name,
                cache_dir=self._get_model_cache_dir(),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )

            model = model.to(device)
            model.config.use_cache = True
            model.eval()

            # Matmul perf hint (harmless on CPU)
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

            self.logger.info(f"Loaded '{self._model_name}' on {device} (dtype={dtype})")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load summarization model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load summarization model: {e}")

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load FLAN-T5 tokenizer; ensure PAD token is set."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                cache_dir=self._get_model_cache_dir(),
                use_fast=True,
            )
            # T5 has a dedicated <pad>; keep it (don’t overwrite with eos)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
            self.logger.debug(f"Loaded tokenizer for {self._model_name}")
            return tokenizer
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise RuntimeError(f"Failed to load tokenizer: {e}")

    def _get_model_cache_dir(self) -> str:
        return self.model_cache_dir

    def get_generation_config(self) -> GenerationConfig:
        """
        Generation tuned for T5:
        - Small beam size for quality vs latency.
        - Deterministic (no sampling) for reproducibility.
        - Mild anti-repetition.
        """
        if self._gen_config is None:
            self._gen_config = GenerationConfig(
                num_beams=4,
                length_penalty=1.1,       # slightly favors concise text
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,  # light touch for T5
                do_sample=False,
            )
        return self._gen_config

    # --- Instruction for robust news cleanup/structuring ---
    def _build_instruction_prompt(self, text: str) -> str:
        """
        Instruction focuses on removing boilerplate (ads/cookies/placeholders),
        dropping off-topic politics/crime if unrelated, and producing a clean,
        coherent event-focused summary.
        """
        return (
            "Task: Clean and summarize the following news text.\n"
            "- Remove ads, cookie notices, trackers, boilerplate, placeholders.\n"
            "- Drop unrelated politics/crime if off-topic to the main event.\n"
            "- Output 2–4 concise sentences focusing only on the described event.\n\n"
            f"Text:\n{text}"
        )

    def summarize_text(
        self,
        text: str,
        instance: Optional[ModelInstance[AutoModelForSeq2SeqLM]] = None
    ) -> str:
        """
        Summarize/clean text using FLAN-T5-Base with an instruction prompt.
        """
        if not text or not text.strip():
            return ""
        if instance is None:
            with self.acquire() as instance:
                return self._perform_summarization(text, instance)
        else:
            return self._perform_summarization(text, instance)

    def _perform_summarization(
        self,
        text: str,
        instance: ModelInstance[AutoModelForSeq2SeqLM]
    ) -> str:
        """FLAN-T5 generation with instruction prompt + dynamic lengths."""
        try:
            tokenizer = instance.tokenizer
            model = instance.model
            device = instance.device

            prompt = self._build_instruction_prompt(text)

            # Respect model max length; keep safety headroom
            model_max = getattr(tokenizer, "model_max_length", 512)
            in_max = min(self._max_tokens, max(128, model_max - 32))

            inputs = tokenizer(
                prompt,
                max_length=in_max,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            gc = self.get_generation_config()

            # Token-aware dynamic new-token budget (based on input tokens)
            n_in_tokens = int(inputs.input_ids.shape[1])
            gc_max = max(80, min(180, n_in_tokens // 2))
            gc_min = min(64, max(0, gc_max // 3))
            # ensure consistency even if HF mutates config object
            setattr(gc, "max_new_tokens", gc_max)
            setattr(gc, "min_new_tokens", gc_min)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    generation_config=gc,
                )

            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.strip()

        except Exception as e:
            self.logger.error(f"Failed to summarize text: {e}")
            raise RuntimeError(f"Summarization failed: {e}")
