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

import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional
from app.core.exceptions import ServiceException
from app.core.concurrency import InferenceRuntime, RuntimeConfig
from app.nlp import Translator, NLPUtils
from app.config import get_service_logger, get_settings
from app.etl_data.etl_models import FeedModel
from app.core.model import SummarizationModelManager
from app.core.concurrency import CPUExecutors
from app.enumeration import WorkloadEnums

class SummarizerTask:
    """
    Summarizes raw article texts using a BART model (facebook/bart-large-cnn).
    For the prod env:
    torch==2.3.0+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html
    transformers
    (Replace cu118 with cu121 or cu117 if needed.)
    For the dev env:
    torch==2.3.0+cpu
    transformers
    """

    def __init__(self):
        self.logger = get_service_logger("Summarizer")
        self.settings = get_settings()
        # inference runtime for summarization
        self.inference_runtime: Optional[InferenceRuntime] = None
        self.cpu_executors = CPUExecutors(workload=WorkloadEnums.CPU)
        
        
    def get_summarizer_utils():
        """Lazy import to avoid circular dependency."""
        from app.etl.tasks import SummarizerUtils

        return SummarizerUtils

    async def summarize_all_feeds(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """
        Translate, compress if needed, and summarize each article using InferenceRuntime with GPU micro-batching.
        """
        try:
            self.logger.info(f"Summarizer: summarizing {len(feeds)} feeds")
            
            # Initialize inference runtime if not already done
            if not self.inference_runtime:
                self.logger.info("Summarizer: initializing inference runtime")
                await self._initialize_inference_runtime()

            summarizer: SummarizationModelManager = self.inference_runtime.model_manager
            batch_size = summarizer.pool_size
            summarized_feeds: list[FeedModel] = []

            # Process feeds in batches
            self.logger.info(f"Summarizer: processing {len(feeds)} feeds in batches of {batch_size}")
            for i in range(0, len(feeds), batch_size):
                batch = feeds[i : i + batch_size]
                self.logger.info(f"Summarizer: processing batch of {len(batch)} feeds")
                batch_feeds = await self._process_batch(batch) # parallel execution of the batch
                summarized_feeds.extend(batch_feeds)               
                

            return summarized_feeds

        except Exception as e:
            self.logger.error(f"Summarizer: error summarizing feeds: {e}")
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
                model_manager_loader=self._get_summarization_model_manager, config=config
            )

            await self.inference_runtime.start()
            self.logger.info(
                "summarizer.py:Summarizer:Inference runtime initialized for summarization"
            )

        except Exception as e:
            self.logger.error(
                f"summarizer.py:Summarizer:Failed to initialize inference runtime: {e}"
            )
            raise

    def _get_summarization_model_manager(self):
        """Model loader function for the inference runtime."""
        # Return only the model, not the tuple, since GPUWorker expects a single model        
        return SummarizationModelManager(self.settings)
    

    async def _process_batch(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """Process a batch of feeds using the inference runtime with CPI/GPU micro-batching."""
        try:
            from app.etl.tasks import SummarizerUtils
            summarizer_utils = SummarizerUtils()

            summarizer: SummarizationModelManager = self.inference_runtime.model_manager

            async def run(feed: FeedModel) -> FeedModel | None:
                # each task acquires its own instance
                async with summarizer.acquire(destroy_after_use=False) as instance:
                    try:
                        # result = summarizer_utils._summarizer(
                        #     feed.processed_text,
                        #     instance.model,
                        #     instance.tokenizer,
                        #     instance.device,
                        #     instance.max_tokens,
                        # )
                        
                        result = await self.cpu_executors.run_in_thread(
                             summarizer_utils._summarizer,  
                             feed.processed_text, 
                             instance.model, 
                             instance.tokenizer, 
                             instance.device, 
                             instance.max_tokens
                        )
                        if result is None:
                            raise Exception("Summarizer: Error summarizing feed")
                        
                        feed.processed_text = result
                        feed.summary = result
                        return feed
                    except Exception as e:
                        self.logger.error(f"Summarizer: Error summarizing feed: {e}")
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
            self.logger.error(f"Summarizer: Batch processing failed: {e}")
            return []       
        

    # ─── Generate Questions from Summary ──────────────────────────────────────────────On trial for now
    def generate_questions_from_summary(self, feeds: list[FeedModel], max_questions: int = 3) -> list:
        """
        Generate up to 3 concise questions from a given summary using Flan-T5-Large.
        """

        # Load the tokenizer and model from Hugging Face Hub
        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        for feed in feeds:

            prompt = f"Generate {max_questions} questions based on this summary:\n{feed.summary}"

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(model.device)

            # Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Split into lines, return up to `max_questions` questions
            questions = [
                q.strip("- ").strip() for q in generated_text.split("\n") if q.strip()
            ]
            feed.questions = questions
