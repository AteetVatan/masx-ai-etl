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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional
from app.core.exceptions import ServiceException
from app.core.concurrency import InferenceRuntime, RuntimeConfig
from app.nlp import Translator, NLPUtils
from app.singleton import ModelManager
from app.config import get_service_logger, get_settings
from app.etl_data.etl_models import FeedModel
from app.core.models import SummarizationFinalizerModelManager
from app.core.concurrency import CPUExecutors
from app.enumeration import WorkloadEnums

class SummarizerFinalizerTask:
    """
    Summarizes raw article texts using a FLAN-T5-Base model (google/flan-t5-base).
    """

    def __init__(self):
        self.logger = get_service_logger("SummarizerFinalizer")
        self.settings = get_settings()
        # inference runtime for summarization
        self.inference_runtime: Optional[InferenceRuntime] = None
        self.cpu_executors = CPUExecutors(workload=WorkloadEnums.CPU)
        
        
    def get_summarizer_finalizer_utils():
        """Lazy import to avoid circular dependency."""
        from app.etl.tasks import SummarizerFinalizerUtils

        return SummarizerFinalizerUtils

    async def summarize_all_feeds_finalizer(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """
        Translate, compress if needed, and summarize each article using InferenceRuntime with GPU micro-batching.
        """
        try:
            self.logger.info(f"SummarizerFinalizer: summarizing {len(feeds)} feeds")
            
            # Initialize inference runtime if not already done
            if not self.inference_runtime:
                self.logger.info("SummarizerFinalizer: initializing inference runtime")
                await self._initialize_inference_runtime()

            finalizer: SummarizationFinalizerModelManager = self.inference_runtime.model_manager
            batch_size = finalizer.pool_size
            finalized_feeds: list[FeedModel] = []

            # Process feeds in batches
            for i in range(0, len(feeds), batch_size):
                batch = feeds[i : i + batch_size]
                self.logger.info(f"SummarizerFinalizer: processing batch of {len(batch)} feeds")
                batch_feeds = await self._process_batch(batch) # parallel execution of the batch
                finalized_feeds.extend(batch_feeds)

            return finalized_feeds

        except Exception as e:
            self.logger.error(f"SummarizerFinalizer: error summarizing feeds: {e}")
            raise ServiceException(f"Error summarizing feeds: {e}")
        finally:
            # Final cleanup -- remove all the models from the pool
            if self.inference_runtime:
                self.inference_runtime.model_manager.cleanup()
                #await self.inference_runtime.stop()
            if self.cpu_executors:
                self.cpu_executors.shutdown(wait=True)
                

    async def _initialize_inference_runtime(self):
        """Initialize the inference runtime for summarization."""
        try:
            # Create runtime config optimized for summarization
            config = RuntimeConfig(
            )

            # Create and start inference runtime
            self.inference_runtime = InferenceRuntime(
                model_manager_loader=self._get_finalizer_model_manager, config=config
            )

            await self.inference_runtime.start()
            self.logger.info(
                "summarizer.py:SummarizerFinalizer:Inference runtime initialized for summarization"
            )

        except Exception as e:
            self.logger.error(
                f"summarizer.py:SummarizerFinalizer:Failed to initialize inference runtime: {e}"
            )
            raise

    def _get_finalizer_model_manager(self):
        """Model loader function for the inference runtime."""
        # Return only the model, not the tuple, since GPUWorker expects a single model        
        return SummarizationFinalizerModelManager(self.settings)
    

    async def _process_batch(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """Process a batch of feeds using the inference runtime with CPI/GPU micro-batching."""
        try:
            from app.etl.tasks import SummarizerFinalizerUtils
            summarizer_utils = SummarizerFinalizerUtils()
           
            finalizer: SummarizationFinalizerModelManager = self.inference_runtime.model_manager

            async def run(feed: FeedModel) -> FeedModel | None:
                # each task acquires its own instance
                async with finalizer.acquire(destroy_after_use=False) as instance:
                    try:
                        # result = summarizer_utils._summarizer(
                        #     feed.processed_text,
                        #     instance.model,
                        #     instance.tokenizer,
                        #     instance.device,
                        #     instance.max_tokens,
                        # )
                        
                        result = await self.cpu_executors.run_in_thread(
                             summarizer_utils._summarizer_finalizer,  
                             feed.processed_text, 
                             instance.model, 
                             instance.tokenizer, 
                             instance.device, 
                             instance.max_tokens
                        )
                        if result is None:
                            raise Exception("SummarizerFinalizer: Error summarizing feed")
                        
                        feed.processed_text = result
                        feed.summary = result
                        return feed
                    except Exception as e:
                        self.logger.error(f"SummarizerFinalizer: Error summarizing feed: {e}")
                        return None

            # schedule all feeds in parallel
            tasks = [run(feed) for feed in feeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)   
              
            # filter out None values and exceptions
            summarized_feeds = [
                feed for feed in results if isinstance(feed, FeedModel)
            ]
            return summarized_feeds

        except Exception as e:
            self.logger.error(f"SummarizerFinalizer: Batch processing failed: {e}")
            return [] 