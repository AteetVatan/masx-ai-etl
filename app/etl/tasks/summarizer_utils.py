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


import logging
import asyncio
from app.etl_data.etl_models import FeedModel


class SummarizerUtils:

    logger = logging.getLogger(__name__)

    @staticmethod
    def get_nlp_utils():
        """Lazy import to avoid circular dependency."""
        from app.nlp import NLPUtils

        return NLPUtils

    @staticmethod
    def get_translator():
        """Lazy import to avoid circular dependency."""
        from app.nlp import Translator

        return Translator

    @staticmethod
    def _summarizer(text: str, model, tokenizer, device, max_tokens) -> dict:
        SummarizerUtils.logger.info(
            f"summarizer_utils.py:_summarizer called with payload"
        )

        nlp_utils = SummarizerUtils.get_nlp_utils()
        Translator = SummarizerUtils.get_translator()

        # before translating compress if token > 1024
        # calculate the total number of tokens in the raw_text
        total_tokens = len(tokenizer.tokenize(text))

        # if total_tokens is less than max_tokens, then summarize directly
        # check and debug this part

        if total_tokens < max_tokens:
            summaries = [text]
            result = SummarizerUtils._final_summary(
                model, tokenizer, device, summaries, max_tokens
            )
            return result

        # now we need to chunk and summarize the compressed text
        # -------- Map–Reduce with overlap --------
        try:
            prompt_skeleton = (
                "Summarize for an analyst. Include who/what/when/where/how, numbers, and new developments. "
                "Avoid vague time words (‘today’, ‘recent’); use absolute dates if present. No speculation.\n\n"
            )
            windows = nlp_utils.chunk_by_tokens_overlap(
                tokenizer,
                text,
                window_tokens=950,
                overlap_tokens=50,
                prompt_prefix=prompt_skeleton,
            )
            chunk_summaries = nlp_utils.summarize_windows(
                model,
                tokenizer,
                device,
                windows,
                model_max_tokens=max_tokens,
                gen_kwargs={
                    "min_length": 250,
                    "max_length": 800,
                    "num_beams": 5,
                    "no_repeat_ngram_size": 3,
                    "repetition_penalty": 1.05,
                },
            )
        except Exception as e:
            SummarizerUtils.logger.error(
                f"runtime.py:Window summarization failed for : {e}"
            )
            # single-shot fall back -- check this part
            chunk_summaries = [
                nlp_utils.summarize_text(
                    model,
                    tokenizer,
                    device,
                    prompt_skeleton + text,
                    max_tokens,
                    {"min_length": 250},
                )
            ]

        # -------- Merge & Polish Final summary --------
        result = SummarizerUtils._final_summary(
            model, tokenizer, device, chunk_summaries, max_tokens
        )

        # -------- Quality gates --------
        # try:
        #     q = nlp_utils.run_quality_gates(final_summary, must_keep, ents)
        #     final_summary = q.pop("summary", final_summary)
        #     #result["quality"] = q
        # except Exception as e:
        #     SummarizerUtils.logger.error(
        #         f"runtime.py:Quality gates failed for {e}"
        #     )
        # final_summary = "\n\n".join(chunk_summaries)
        return result

    @staticmethod
    def _final_summary(model, tokenizer, device, chunk_summaries, max_tokens) -> dict:
        """
        Final summary generation.
        """
        try:
            nlp_utils = SummarizerUtils.get_nlp_utils()
            final_summary = nlp_utils.merge_and_polish(
                model, tokenizer, device, chunk_summaries, model_max_tokens=max_tokens
            )
            return final_summary
        except Exception as e:
            SummarizerUtils.logger.error(f"runtime.py:Merge & polish failed for : {e}")
            final_summary = "\n\n".join(chunk_summaries)
