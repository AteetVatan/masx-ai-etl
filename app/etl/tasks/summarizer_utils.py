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
from app.singleton import ModelManager



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
    def _summarizer(payload: dict, model, tokenizer, device, max_tokens) -> dict:
        SummarizerUtils.logger.info(f"summarizer_utils.py:_summarizer called with payload: {payload}")
        # -------- Extract payload --------
        feed_data = payload.get("feed")
        raw_text = payload.get("text", "")
        url = payload.get("url", "")
        #prompt_prefix = payload.get("prompt_prefix", "summarize: ")

        result = {
            "feed_data": feed_data,
            "text": raw_text,
            "url": url,
            "translated_text": None,
            "compressed_text": None,
            "summary": None,
            "quality": None,
        }
        
        nlp_utils = SummarizerUtils.get_nlp_utils()
        Translator = SummarizerUtils.get_translator()
        
        
        # before translating compress if token > 1024
        # calculate the total number of tokens in the raw_text
        total_tokens = len(tokenizer.tokenize(raw_text))
        lang = ModelManager.detect_language(raw_text)
        
        if total_tokens > 2 * max_tokens or lang != "en":
            #if this is the case, then compress the text
             # -------- NER: must-keep entities --------
            try:               
                lang = ModelManager.detect_language(raw_text)
                ents = nlp_utils.extract_entities(raw_text, lang)
                must_keep = nlp_utils.build_must_keep_entities(ents, top_n=15)
            except Exception as e:
                SummarizerUtils.logger.error(f"runtime.py:NER failed for {url}: {e}")
                ents, must_keep = {"DATE": [], "CARDINAL": []}, []
                
            ratio = total_tokens / max_tokens
            if ratio <= 1.5:
                target_tokens = max_tokens
            else:
                target_tokens = max_tokens * 2
            SummarizerUtils.logger.info(f"target_tokens: {target_tokens}")
          
                
            # -------- Adaptive compression to ~2k tokens (pre-Abstractive) --------
            try:
                # Aim ~2 * encoder window before map-reduce (≈2000 tokens)
                compressed = nlp_utils.compress_news_adaptive(
                    tokenizer,
                    raw_text,
                    model_max_tokens=max_tokens,
                    prompt_prefix="",
                    keep_bounds=(0.2, 0.4),
                    must_keep=must_keep,
                    target_tokens=target_tokens,
                    lang=lang,
                )
                result["compressed_text"] = compressed
            except Exception as e:
                SummarizerUtils.logger.error(f"runtime.py:Adaptive compression failed for {url}: {e}")
                result["compressed_text"] = raw_text
                compressed = raw_text          
                
        else:
            compressed = raw_text
            result["compressed_text"] = compressed
            
        # -------- Preprocess: language + translation --------
        try:
            translator = Translator()
            text_en = translator.ensure_english_sync(result["compressed_text"])
            result["translated_text"] = text_en
        except Exception as e:
            SummarizerUtils.logger.error(f"runtime.py:Translation failed for {url}: {e}")
            text_en = raw_text
            result["translated_text"] = text_en
    
        
        # calculate the total number of tokens in the raw_text
        total_tokens = len(tokenizer.tokenize(text_en))
        # if total_tokens is less than max_tokens, then summarize directly
        if total_tokens < max_tokens * 2:
            summaries = [text_en]    
            result["summary"] = SummarizerUtils._final_summary(model, tokenizer, device, summaries, max_tokens)
            return result
           
        #now we need to chunk and summarize the compressed text        
        # -------- Map–Reduce with overlap --------
        try:
            prompt_skeleton = (
                "Summarize for an analyst. Include who/what/when/where/how, numbers, and new developments. "
                "Avoid vague time words (‘today’, ‘recent’); use absolute dates if present. No speculation.\n\n"
            )
            windows = nlp_utils.chunk_by_tokens_overlap(
                tokenizer,
                text_en,
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
                gen_kwargs={"min_length": 250, "max_length": 800, "num_beams": 5, "no_repeat_ngram_size": 3, "repetition_penalty": 1.05},
            )
        except Exception as e:
            SummarizerUtils.logger.error(f"runtime.py:Window summarization failed for {url}: {e}")
            # single-shot fall back
            chunk_summaries = [nlp_utils.summarize_text(model, tokenizer, device, prompt_skeleton + compressed, max_tokens, {"min_length": 250})]               

            
        # -------- Merge & Polish Final summary --------
        final_summary = SummarizerUtils._final_summary(model, tokenizer, device, chunk_summaries, max_tokens)
          
            
        # -------- Quality gates --------
        try:
            q = nlp_utils.run_quality_gates(final_summary, must_keep, ents)
            final_summary = q.pop("summary", final_summary)
            result["quality"] = q
        except Exception as e:
            SummarizerUtils.logger.error(f"runtime.py:Quality gates failed for {url}: {e}")
            final_summary = "\n\n".join(chunk_summaries)
        
        result["summary"] = final_summary
        return result
        
    @staticmethod
    def _final_summary(model, tokenizer, device, chunk_summaries, max_tokens) -> dict:
        """
        Final summary generation.
        """
        try:
            nlp_utils = SummarizerUtils.get_nlp_utils()
            final_summary = nlp_utils.merge_and_polish(
                model,
                tokenizer,
                device,
                chunk_summaries,
                model_max_tokens=max_tokens
            )
            return final_summary
        except Exception as e:
            SummarizerUtils.logger.error(f"runtime.py:Merge & polish failed for : {e}")
            final_summary = "\n\n".join(chunk_summaries)