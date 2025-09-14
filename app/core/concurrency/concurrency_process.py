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

import asyncio
from typing import Awaitable, Callable, Optional
from app.etl_data.etl_models import FeedModel
from app.enumeration.enums import TaskEnums
from app.core.exceptions import ServiceException
from app.core.concurrency.runtime import InferenceRuntime, RuntimeConfig
from app.singleton import ModelManager



from app.config import get_service_logger, get_settings
class ConcurrencyProcess:
    
    def __init__(self, task: TaskEnums, task_executor: Callable, feeds: list[FeedModel]):
        self.logger = get_service_logger("ConcurrencyUtils")
        self.settings = get_settings()
        self.feeds = feeds
        self.task = task
        self.task_executor = task_executor
        self.inference_runtime: Optional[InferenceRuntime] = None
        
    async def initialize_inference_runtime(self):
        """Initialize the inference runtime for summarization."""
        try:
            # Create runtime config optimized for summarization
            config = RuntimeConfig(
            )

            # Create and start inference runtime
            self.inference_runtime = InferenceRuntime(
                model_manager_loader=self._get_model_loader, config=config
            )

            await self.inference_runtime.start()
            self.logger.info(
                f"{self.task.value}:ConcurrencyUtils:Inference runtime initialized for summarization"
            )

        except Exception as e:
            self.logger.error(
                f"{self.task.value}:ConcurrencyUtils:Failed to initialize inference runtime: {e}"
            )
            raise
        
        
    def _get_model_loader(self):
        """Model loader function for the inference runtime."""
        if self.task == TaskEnums.SUMMARIZER:
            return self._get_summarization_model_loader()
        elif self.task == TaskEnums.TRANSLATOR:
            return self._get_translator_model_loader()
        else:
            raise ValueError(f"Invalid task: {self.task}")

    def _get_summarization_model_loader(self):
        """Model loader function for the inference runtime."""
        # Return only the model, not the tuple, since GPUWorker expects a single model
        model, tokenizer, device = ModelManager.get_summarization_model()
        return model
    
    
    def _get_translator_model_loader(self):
        """Model loader function for the inference runtime."""
        model, tokenizer, device = ModelManager.get_translator_model()
        return model
    
    
    def calculate_optimal_concurrency(self) -> int:
        """
        Calculate optimal concurrency based on available resources and configuration.
        
        Returns:
            Optimal number of concurrent batch processors
        """
        try:
            # Base concurrency on batch size and system resources
            base_concurrency = max(1, len(self.feeds) // self.settings.summarizer_batch_size)
            
            # GPU path: Higher concurrency due to micro-batching
            if hasattr(self.inference_runtime, 'use_gpu_flag') and self.inference_runtime.use_gpu_flag:
                # GPU can handle more concurrent batches due to micro-batching
                gpu_concurrency = min(base_concurrency * 2, 8)  # Cap at 8 concurrent batches
                self.logger.info(f"{self.task.value}:ConcurrencyUtils:GPU concurrency: {gpu_concurrency}")
                return gpu_concurrency
            
            # CPU path: Conservative concurrency to avoid resource exhaustion
            cpu_concurrency = min(base_concurrency, 4)  # Cap at 4 concurrent batches for CPU
            self.logger.info(f"{self.task.value}:ConcurrencyUtils:CPU concurrency: {cpu_concurrency}")
            return cpu_concurrency
            
        except Exception as e:
            self.logger.warning(f"{self.task.value}:ConcurrencyUtils:Error calculating concurrency, using default: {e}")
            return 2  # Safe default
        
    def create_optimal_batches(self) -> list[list[FeedModel]]:
        """
        Create optimal batches for parallel processing.
        
        Returns:
            List of feed batches optimized for parallel processing
        """
        try:
            batch_size = self.settings.summarizer_batch_size
            batches = []
            
            # Create batches with optimal sizing
            for i in range(0, len(self.feeds), batch_size):
                batch = self.feeds[i:i + batch_size]
                batches.append(batch)
            
            # Optimize last batch if it's too small
            if len(batches) > 1 and len(batches[-1]) < batch_size // 2:
                # Merge last batch with second-to-last for better GPU utilization
                batches[-2].extend(batches[-1])
                batches.pop()
                self.logger.info(f"{self.task.value}:ConcurrencyUtils:Optimized last batch, total batches: {len(batches)}")
            
            return batches
            
        except Exception as e:
            self.logger.error(f"{self.task.value}:ConcurrencyUtils:Error creating batches: {e}")
            # Fallback: single batch
            return [self.feeds]
        
    async def process_batches_sequential(self, batches: list[list[FeedModel]]) -> list[FeedModel]:
        """
        Process batches sequentially as fallback.
        
        Args:
            batches: List of feed batches
            
        Returns:
            List of processed feeds
        """
        try:
            self.logger.info(f"{self.task.value}:ConcurrencyUtils:Processing {len(batches)} batches sequentially")
            feeds = []
            
            for i, batch in enumerate(batches):
                try:
                    batch_result = await self.process_batch(batch)
                    if batch_result:
                        feeds.extend(batch_result)
                    self.logger.debug(f"{self.task.value}:ConcurrencyUtils:Processed batch {i+1}/{len(batches)}")
                except Exception as e:
                    self.logger.error(f"{self.task.value}:ConcurrencyUtils:Sequential batch {i} failed: {e}")
                    # Continue with next batch
                    continue
            
            return feeds
            
        except Exception as e:
            self.logger.error(f"{self.task.value}:ConcurrencyUtils:Sequential batch processing failed: {e}")
            raise
        
        
    async def process_batch_sequential(
        self, feeds: list[FeedModel]
    ) -> list[FeedModel]:
        """Fallback sequential processing."""
        results = []
        for feed in feeds:
            try:
                processed_feed = await self._runner(self.task_executor, feed)
                if processed_feed:
                    results.append(processed_feed)
            except Exception as e:
                self.logger.error(
                    f"{self.task.value}:ConcurrencyUtils:Feed processing failed: {e}"
                )
        return results
    
    async def process_batches_parallel(self, batches: list[list[FeedModel]], concurrency: int) -> list[FeedModel]:
        """
        Process batches in parallel with pipeline parallelism for maximum performance.
        
        Args:
            batches: List of feed batches
            concurrency: Number of concurrent processors
            
        Returns:
            List of processed feeds
        """
        try:
            # Create semaphore to limit concurrent batch processing
            semaphore = asyncio.Semaphore(concurrency)
            
            # Create tasks for parallel batch processing
            tasks = []
            for batch in batches:
                task = self.process_batch_with_semaphore(semaphore, batch)
                tasks.append(task)
            
            # Execute all tasks concurrently
            self.logger.info(f"{self.task.value}:ConcurrencyUtils:Processing {len(tasks)} batches with {concurrency} concurrent processors")
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect and flatten results
            processed_feeds = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"{self.task.value}:ConcurrencyUtils:Batch {i} failed: {result}")
                    # Process failed batch sequentially as fallback
                    try:
                        fallback_result = await self.process_batch(batches[i])
                        if fallback_result:                            
                            processed_feeds.extend(fallback_result)
                    except Exception as fallback_error:
                        self.logger.error(f"{self.task.value}:ConcurrencyUtils:Fallback for batch {i} also failed: {fallback_error}")
                elif result:
                    processed_feeds.extend(result)
            
            self.logger.info(f"{self.task.value}:ConcurrencyUtils:Successfully processed {len(processed_feeds)} feeds from {len(batches)} batches")
            return processed_feeds
            
        except Exception as e:
            self.logger.error(f"{self.task.value}:ConcurrencyUtils:Parallel batch processing failed: {e}")
            # Fallback to sequential processing
            return await self.process_batches_sequential(batches)
        
    
    async def process_batch_async(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """Process a batch of feeds using async concurrency."""
        try:
            # Create tasks for concurrent processing
            #tasks = [self._summarize_feed_async(feed) for feed in feeds]
            
            tasks = [self._runner(self.task_executor, feed) for feed in feeds]           
            

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            processed_feeds = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(
                        f"summarizer.py:Summarizer:Feed {i} processing failed: {result}"
                    )
                    continue

                if result:
                    processed_feeds.append(result)

            return processed_feeds

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            # Fallback to sequential processing
            try:
                return await self.process_batch_sequential(feeds)
            except Exception as fallback_error:
                self.logger.error(f"Sequential fallback also failed: {fallback_error}")
                # Last resort: return empty list to prevent complete failure
                return []
            
    
    async def process_batch_with_semaphore(self, semaphore: asyncio.Semaphore, batch: list[FeedModel]) -> list[FeedModel]:
        """
        Process a single batch with semaphore-controlled concurrency.
        Concurrent programming (multithreading, multiprocessing, or async systems) to control access to a shared resource.
        
        Args:
            semaphore: Concurrency control semaphore
            batch: Batch of feeds to process
            
        Returns:
            List of processed feeds
        """
        async with semaphore:
            try:
                return await self.process_batch(batch)
            except Exception as e:
                self.logger.error(f"summarizer.py:Summarizer:Semaphore-controlled batch processing failed: {e}")
                raise
            
    async def process_batch(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """Process a batch of feeds using the inference runtime with GPU micro-batching."""
        try:
            # Prepare payloads for inference
            payloads = []
            for feed in feeds:
                payload = {
                    "feed": feed,
                    "text": feed.raw_text,
                    "text_en": feed.raw_text_en,
                    "compressed_text": feed.compressed_text,
                    "translated_text": feed.raw_text_en,
                    "summary": feed.summary,
                    "url": feed.url,
                    "prompt_prefix": self.prompt_prefix,
                }
                payloads.append(payload)

            # Use inference runtime for batch processing with GPU micro-batching
            self.logger.info(
                f"{self.task.value}:ConcurrencyUtils:process_batch using inference runtime"
            )            
            
            
            if self.task == TaskEnums.SUMMARIZER:           
                results = await self.inference_runtime.infer_many(payloads)
            elif self.task == TaskEnums.TRANSLATOR:                
                #results = await self.inference_runtime.infer_many(payloads)
                try:
                    tasks = [self.task_executor(feed) for feed in feeds]
                    results = await asyncio.gather(*tasks)
                    return results
                except Exception as e:
                    self.logger.error(f"{self.task.value}:TranslatorTask:Error translating feeds: {e}")
                    raise ServiceException(f"Error translating feeds: {e}")
            
            

            # Process results
            processed_feeds = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(
                        f"{self.task.value}:ConcurrencyUtils:Feed {i} processing failed: {result}"
                    )
                    # Fallback to direct summarization
                    try:
                        processed_feed = await self._runner(self.task_executor, feed)
                        if processed_feed:
                            processed_feeds.append(processed_feed)
                    except Exception as e:
                        self.logger.error(
                            f"{self.task.value}:ConcurrencyUtils:Direct summarization also failed for feed {i}: {e}"
                        )
                    continue

                # Apply the processed result to the feed
                feed = feeds[i]
                if result and isinstance(result, dict):
                    # Update feed with processed result from inference runtime
                    if "translated_text" in result:
                        feed.raw_text_en = result["translated_text"]
                    if "compressed_text" in result:
                        feed.compressed_text = result["compressed_text"]
                    if "summary" in result:
                        feed.summary = result["summary"]

                    # Only add if we got a valid summary
                    if hasattr(feed, "summary") and feed.summary:
                        processed_feeds.append(feed)
                    else:
                        # Fallback to direct summarization if no summary was generated
                        try:
                            processed_feed = await self._runner(self.task_executor, feed)
                            if processed_feed:
                                processed_feeds.append(processed_feed)
                        except Exception as e:
                            self.logger.error(
                                f"{self.task.value}:ConcurrencyUtils:Fallback summarization failed for feed {i}: {e}"
                            )
                elif result and isinstance(result, Exception):
                    # Handle case where result is an exception
                    self.logger.error(
                        f"{self.task.value}:ConcurrencyUtils:Inference runtime returned exception for feed {i}: {result}"
                    )
                    # Fallback to direct summarization
                    try:
                        processed_feed = await self._runner(self.task_executor, feed)
                        if processed_feed:
                            processed_feeds.append(processed_feed)
                    except Exception as e:
                        self.logger.error(
                            f"{self.task.value}:ConcurrencyUtils:Fallback summarization also failed for feed {i}: {e}"
                        )

            return processed_feeds

        except Exception as e:
            self.logger.error(f"{self.task.value}:ConcurrencyUtils:Batch processing failed: {e}")
            # Fallback to sequential processing
            try:
                return await self.process_batch_sequential(feeds)
            except Exception as fallback_error:
                self.logger.error(
                    f"{self.task.value}:ConcurrencyUtils:Sequential fallback also failed: {fallback_error}"
                )
                # Last resort: return empty list to prevent complete failure
                return []
            
    async def process_feeds_parallel(self, concurrency: int) -> list[FeedModel]:
        """
        Process feeds using high-performance parallel processing with pipeline parallelism.
        
        Args:
            concurrency: Number of concurrent batch processors
            
        Returns:
            List of processed feeds
        """
        try:
            # Create batches for parallel processing
            batches = self.create_optimal_batches()
            self.logger.info(f"{self.task.value}:ConcurrencyUtils:Created {len(batches)} batches for parallel processing")
            
            # Process batches in parallel with pipeline parallelism
            if concurrency == 1:
                # Single-threaded fallback for debugging
                return await self.process_batches_sequential(batches)
            else:
                # High-performance parallel processing
                return await self.process_batches_parallel(batches, concurrency)
                
        except Exception as e:
            self.logger.error(f"{self.task.value}:ConcurrencyUtils:Parallel processing failed: {e}")
            # Fallback to sequential processing
            return await self.process_feeds_sequential()
            
    async def process_feeds_sequential(self) -> list[FeedModel]:
        """
        Fallback to sequential processing if parallel processing fails.
        
        Returns:
            List of processed feeds
        """
        try:
            self.logger.warning(f"{self.task.value}:ConcurrencyUtils:Falling back to sequential processing")
            processed_feeds = []
            
            for i, feed in enumerate(self.feeds):
                try:
                    processed_feed = await self._runner(self.task_executor, feed)
                    
                    if processed_feed:
                        processed_feeds.append(processed_feed)
                    
                    # Log progress every 10 feeds
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"{self.task.value}:ConcurrencyUtils:Sequential progress: {i+1}/{len(self.feeds)}")
                        
                except Exception as e:
                    self.logger.error(f"{self.task.value}:ConcurrencyUtils:Sequential feed {i} failed: {e}")
                    continue
            
            return processed_feeds
            
        except Exception as e:
            self.logger.error(f"{self.task.value}:ConcurrencyUtils:Sequential processing failed: {e}")
            raise

    #       ___________ Runner Function ________________
    async def _runner(func: Callable[[FeedModel], Awaitable[FeedModel]], arg: FeedModel):
        print("Running:", await func(arg))      