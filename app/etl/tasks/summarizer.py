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

from app.core.exceptions import ServiceException
from app.core.concurrency import InferenceRuntime, RuntimeConfig
from app.nlp import Translator, NLPUtils
from app.singleton import ModelManager
from app.config import get_service_logger, get_settings
from app.etl_data.etl_models import FeedModel
from typing import Optional


class Summarizer:
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

    def __init__(self, feeds: list[FeedModel]):
        self.feeds = feeds
        self.summarization_model, self.summarization_tokenizer, self.device = (
            ModelManager.get_summarization_model()
        )
        self.translator = Translator()
        self.logger = get_service_logger("Summarizer")
        self.settings = get_settings()
        self.prompt_prefix = "summarize: "

        # Initialize inference runtime
        self.inference_runtime: Optional[InferenceRuntime] = None

    async def summarize_all_feeds(self):
        """
        Translate, compress if needed, and summarize each article using InferenceRuntime with GPU micro-batching.
        """
        try:
            self.logger.info(f"Summarizer: Summarizing {len(self.feeds)} feeds")
            # Initialize inference runtime if not already done
            if not self.inference_runtime:
                self.logger.info(f"Summarizer: Initializing inference runtime")
                await self._initialize_inference_runtime()

            self.logger.info(f"Inference runtime initialized")
            # Process feeds using the inference runtime
            summarized_feeds = []

            # Process in batches for efficiency
            self.logger.info(f"Summarizer: Process in batches for efficiency")
            batch_size = 10
            for i in range(0, len(self.feeds), batch_size):
                batch = self.feeds[i : i + batch_size]
                batch_results = await self._process_batch(batch)
                summarized_feeds.extend([r for r in batch_results if r])

            return summarized_feeds

        except Exception as e:
            self.logger.error(f"[Summarizer] Error summarizing feeds: {e}")
            raise ServiceException(f"Error summarizing feeds: {e}")
        finally:
            # Cleanup inference runtime
            if self.inference_runtime:
                await self.inference_runtime.stop()

    async def _initialize_inference_runtime(self):
        """Initialize the inference runtime for summarization."""
        try:
            # Create runtime config optimized for summarization
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
                model_loader=self._get_summarization_model_loader, config=config
            )

            await self.inference_runtime.start()
            self.logger.info("Inference runtime initialized for summarization")

        except Exception as e:
            self.logger.error(f"Failed to initialize inference runtime: {e}")
            raise

    def _get_summarization_model_loader(self):
        """Model loader function for the inference runtime."""
        # Return only the model, not the tuple, since GPUWorker expects a single model
        model, tokenizer, device = ModelManager.get_summarization_model()
        return model

    async def _process_batch(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """Process a batch of feeds using the inference runtime with GPU micro-batching."""
        try:
            # Prepare payloads for inference
            payloads = []
            for feed in feeds:
                payload = {
                    "feed": feed,
                    "text": feed.raw_text,
                    "url": feed.url,
                    "prompt_prefix": self.prompt_prefix,
                }
                payloads.append(payload)

            # Use inference runtime for batch processing with GPU micro-batching
            results = await self.inference_runtime.infer_many(payloads)

            # Process results
            processed_feeds = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Feed {i} processing failed: {result}")
                    # Fallback to direct summarization
                    try:
                        processed_feed = await self._summarize_feed_async(feeds[i])
                        if processed_feed:
                            processed_feeds.append(processed_feed)
                    except Exception as e:
                        self.logger.error(
                            f"Direct summarization also failed for feed {i}: {e}"
                        )
                    continue

                # Apply the processed result to the feed
                feed = feeds[i]
                if result and isinstance(result, dict):
                    # Update feed with processed result from inference runtime
                    if "translated_text" in result:
                        feed.raw_text_en = result["translated_text"]
                    if "compressed_text" in result:
                        feed.raw_text_en = result["compressed_text"]
                    if "summary" in result:
                        feed.summary = result["summary"]

                    # Only add if we got a valid summary
                    if hasattr(feed, "summary") and feed.summary:
                        processed_feeds.append(feed)
                    else:
                        # Fallback to direct summarization if no summary was generated
                        try:
                            processed_feed = await self._summarize_feed_async(feed)
                            if processed_feed:
                                processed_feeds.append(processed_feed)
                        except Exception as e:
                            self.logger.error(
                                f"Fallback summarization failed for feed {i}: {e}"
                            )
                elif result and isinstance(result, Exception):
                    # Handle case where result is an exception
                    self.logger.error(
                        f"Inference runtime returned exception for feed {i}: {result}"
                    )
                    # Fallback to direct summarization
                    try:
                        processed_feed = await self._summarize_feed_async(feed)
                        if processed_feed:
                            processed_feeds.append(processed_feed)
                    except Exception as e:
                        self.logger.error(
                            f"Fallback summarization also failed for feed {i}: {e}"
                        )

            return processed_feeds

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            # Fallback to sequential processing
            try:
                return await self._process_batch_sequential(feeds)
            except Exception as fallback_error:
                self.logger.error(f"Sequential fallback also failed: {fallback_error}")
                # Last resort: return empty list to prevent complete failure
                return []

    async def _process_batch_async(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """Process a batch of feeds using async concurrency."""
        try:
            # Create tasks for concurrent processing
            tasks = [self._summarize_feed_async(feed) for feed in feeds]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            processed_feeds = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Feed {i} processing failed: {result}")
                    continue

                if result:
                    processed_feeds.append(result)

            return processed_feeds

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            # Fallback to sequential processing
            try:
                return await self._process_batch_sequential(feeds)
            except Exception as fallback_error:
                self.logger.error(f"Sequential fallback also failed: {fallback_error}")
                # Last resort: return empty list to prevent complete failure
                return []

    async def _process_batch_sequential(
        self, feeds: list[FeedModel]
    ) -> list[FeedModel]:
        """Fallback sequential processing."""
        results = []
        for feed in feeds:
            try:
                result = await self._summarize_feed_async(feed)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Feed processing failed: {e}")
        return results

    async def _summarize_feed_async(self, feed: FeedModel):
        """
        Summarize a single feed asynchronously.
        """
        try:
            # Step 1: Translate non-English articles to English
            try:
                feed.raw_text_en = self.translator.ensure_english(feed.raw_text)
            except Exception as e:
                self.logger.error(f"[Summarizer] Error translating feed: {e}")
                raise ServiceException(f"Error translating feed: {e}")

            # Step 2: Check if text fits the model, else compress using TF-IDF
            try:
                # tokenizer = self.summarization_tokenizer
                tokenizer = self.summarization_tokenizer.__class__.from_pretrained(
                    self.summarization_tokenizer.name_or_path
                )
                max_tokens = ModelManager.get_summarization_model_max_tokens()
                if not NLPUtils.text_suitable_for_model(
                    tokenizer,
                    feed.raw_text_en,
                    max_tokens,
                ):
                    self.logger.info(f"[Summarizer] Compressing text using TF-IDF")
                    text = NLPUtils.compress_text_tfidf(
                        tokenizer, feed.raw_text_en, max_tokens
                    )
                    feed.raw_text_en = text
            except Exception as e:
                self.logger.error(f"[Summarizer] Error compressing text: {e}")
                raise ServiceException(f"Error compressing text: {e}")

            # Step 3: Generate summary
            try:
                summary = self._generate_summary(feed.raw_text_en)
                feed.summary = summary
                self.logger.info(f"[Summarizer] Summary generated for feed: {feed.url}")
            except Exception as e:
                self.logger.error(f"[Summarizer] Error generating summary: {e}")
                raise ServiceException(f"Error generating summary: {e}")

            return feed

        except Exception as e:
            self.logger.error(f"[Summarizer] Error summarizing feed: {e}")
            raise ServiceException(f"Error summarizing feed: {e}")

    def __summarize_feed(self, feed: FeedModel):
        """
        Summarize a single feed (synchronous version for backward compatibility).
        """
        try:

            # Step 1: Translate non-English articles to English
            try:
                feed.raw_text_en = self.translator.ensure_english(feed.raw_text)
            except Exception as e:
                self.logger.error(f"[Summarizer] Error translating feed: {e}")
                raise ServiceException(f"Error translating feed: {e}")

            # Step 2: Check if text fits the model, else compress using TF-IDF
            try:
                # tokenizer = self.summarization_tokenizer
                tokenizer = self.summarization_tokenizer.__class__.from_pretrained(
                    self.summarization_tokenizer.name_or_path
                )
                max_tokens = ModelManager.get_summarization_model_max_tokens()
                if not NLPUtils.text_suitable_for_model(
                    tokenizer,
                    feed.raw_text_en,
                    max_tokens,
                ):
                    self.logger.info(f"[Summarizer] Compressing text using TF-IDF")
                    text = NLPUtils.compress_text_tfidf(
                        tokenizer,
                        feed.raw_text_en,
                        max_tokens,
                        prompt_prefix=self.prompt_prefix,
                    )
                else:
                    text = feed.raw_text_en
            except Exception as e:
                self.logger.error(f"[Summarizer] Error compressing text: {e}")
                raise ServiceException(f"Error compressing text: {e}")

            # Step 3: Summarize using the BART model
            try:
                self.logger.info(f"[Summarizer] Summarizing text using BART model")
                max_tokens = ModelManager.get_summarization_model_max_tokens()
                summarizer = self.summarization_model
                tokenizer = self.summarization_tokenizer.__class__.from_pretrained(
                    self.summarization_tokenizer.name_or_path
                )
                device = self.device
                feed.summary = NLPUtils.summarize_text(
                    summarizer, tokenizer, device, self.prompt_prefix + text, max_tokens
                )
            except Exception as e:
                self.logger.error(f"[Summarizer] Error summarizing text: {e}")
                raise ServiceException(f"Error summarizing text: {e}")

            # step 5: Generate questions from summary
            # self.generate_questions_from_summary(max_questions=3)

            # Step 4: Push serializable version of all NewsArticle objects
            # serialized = [a.model_dump() for a in self.news_articles]
            return feed
        except Exception as e:
            self.logger.error(f"[Summarizer] Error summarizing feed: {e}")
            raise ServiceException(f"Error summarizing feed: {e}")

    # ─── Generate Questions from Summary ──────────────────────────────────────────────On trial for now
    def generate_questions_from_summary(self, max_questions: int = 3) -> list:
        """
        Generate up to 3 concise questions from a given summary using Flan-T5-Large.
        """

        # Load the tokenizer and model from Hugging Face Hub
        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        for feed in self.feeds:

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
