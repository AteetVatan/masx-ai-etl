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

import asyncio
from app.nlp import NLPUtils
from app.config import get_service_logger, get_settings
from app.core.concurrency import CPUExecutors
from app.etl_data.etl_models import FeedModel
from app.core.models import SummarizationModelManager
from app.enumeration import WorkloadEnums

    
class CompressorTask:
    def __init__(self):
        #self.model, self.tokenizer, self.device = ModelManager.get_summarization_model()
        model_manager = SummarizationModelManager()
        self.tokenizer = model_manager.get_tokenizer()
        self.max_tokens = model_manager.max_tokens
        self.nlp_utils = self.get_nlp_utils()
        self.logger = get_service_logger("Compressor")
        self.settings = get_settings()
        self.cpu_executors = CPUExecutors(workload=WorkloadEnums.CPU)
        
        
    async def compress_all_feeds(self, feeds: list[FeedModel]) -> list[FeedModel]:        
        try:
            
            feeds_multilingual = await self._get_feeds_multilingual(feeds)
            
            if len(feeds_multilingual) == 0:
                return feeds
            
            batch_size = self.cpu_executors.max_threads
            results = []
            for i in range(0, len(feeds_multilingual), batch_size):
                batch = feeds_multilingual[i : i + batch_size]
                tasks = [
                    self.cpu_executors.run_in_thread(self._compress_sync, feed)
                    for feed in batch
                ]
                results.extend(await asyncio.gather(*tasks, return_exceptions=True))
             
            # Convert exceptions to logs and filter out broken feeds
            processed: list[FeedModel] = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Feed compression error: {result}")
                else:
                    if result.compressed_text == "": # case when no compression was required
                        result.compressed_text = result.processed_text
                        
                    result.processed_text = result.compressed_text                        
                    processed.append(result)
                    
            for feed in feeds:
                p_feed_item = next((p_feed for p_feed in processed if feed.id == p_feed.id), None)
                if p_feed_item:
                    feed.compressed_text = p_feed_item.compressed_text
                    feed.processed_text = p_feed_item.processed_text
                    
                    
            return feeds            
       
        except Exception as e:
            self.logger.error(f"runtime.py:Compressor:Error compressing feeds: {e}")
            raise
        finally:
            if self.cpu_executors:
                self.cpu_executors.shutdown(wait=True)


    def _compress_sync(self, feed: FeedModel) -> str:
        """Synchronous compression method for thread pool execution."""
        try:
            raw_text = feed.processed_text
            # Calculate tokens
            total_tokens = len(self.tokenizer.tokenize(raw_text))            
            lang = feed.language
            
            if total_tokens > 2 * self.max_tokens or lang != "en":             
                # Extract entities
                try:
                    ents = self.nlp_utils.extract_entities(raw_text, lang)
                    must_keep = self.nlp_utils.build_must_keep_entities(ents, top_n=15)
                except Exception as e:
                    self.logger.error(f"NER failed: {e}")
                    ents, must_keep = {"DATE": [], "CARDINAL": []}, []
                    
                # Calculate target tokens
                ratio = total_tokens / self.max_tokens
                target_tokens = self.max_tokens if ratio <= 1.5 else self.max_tokens * 2
                
                # Adaptive compression
                try:
                    compressed = self.nlp_utils.compress_news_adaptive(
                        self.tokenizer,
                        raw_text,
                        model_max_tokens=self.max_tokens,
                        prompt_prefix="",
                        keep_bounds=(0.2, 0.4),
                        must_keep=must_keep,
                        target_tokens=target_tokens,
                        lang=lang,
                    )
                    feed.compressed_text = compressed
                    return feed
                except Exception as e:
                    self.logger.error(f"Adaptive compression failed: {e}")
                    return feed
            else:
                return feed
                
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            return feed
        
    async def _get_feeds_multilingual(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """
        Return only feeds whose raw_text is not detected as English.
        Uses thread offloading for language detection.
        """
        return [feed for feed in feeds if feed.language != "en"]
        
    @staticmethod
    def get_nlp_utils():
        """Lazy import to avoid circular dependency."""
        from app.nlp import NLPUtils
        return NLPUtils