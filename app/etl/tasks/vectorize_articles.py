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
from typing import List, Optional
from uuid import uuid4
from app.singleton import ModelManager
from app.nlp import VectorDBManager
from app.etl_data.etl_models.feed_model import FeedModel
from app.config import get_service_logger, get_settings
from app.core.concurrency import InferenceRuntime, RuntimeConfig


class VectorizeArticles:
    """ETL Task: Vectorize NewsArticle objects and store them using VectorDBManager."""

    def __init__(self, flashpoint_id: str):
        self.__flashpoint_id = flashpoint_id
        self.__db = VectorDBManager()
        self._logger = get_service_logger("VectorizeArticles")
        self.settings = get_settings()

        # Initialize inference runtime for embedding generation
        self.inference_runtime: Optional[InferenceRuntime] = None

    # gettters
    def get_flashpoint_id(self) -> str:
        return self.__flashpoint_id

    def get_db(self) -> VectorDBManager:
        return self.__db

    async def run(self, feeds: List[FeedModel]) -> str:
        """
        Vectorize and store NewsArticles in Chroma vector DB using async inference runtime.

        Uses `.summary` if available, otherwise falls back to `.raw_text`.

        Returns:
            str: The Chroma collection name used
        """
        try:
            # Initialize inference runtime if not already done
            if not self.inference_runtime:
                await self._initialize_inference_runtime()

            texts, metadatas, ids = [], [], []

            for feed in feeds:
                text = feed.summary.strip()
                if not text:
                    continue

                texts.append(text)
                metadatas.append(
                    {
                        "url": feed.url,
                        "image": feed.image or "unknown",
                        "domain": feed.domain,
                        "sourcecountry": feed.sourcecountry,
                        "language": feed.language or "unknown",
                    }
                )
                ids.append(str(uuid4()))  # each vector gets a unique ID why?

            if not texts:
                self._logger.info("No valid articles to vectorize.")
                return self.__flashpoint_id

            # Embed using inference runtime with micro-batching
            self._logger.info(
                f"Vectorizing {len(texts)} articles into collection: {self.__flashpoint_id}"
            )

            # Prepare payloads for batch processing
            payloads = [{"text": text, "index": i} for i, text in enumerate(texts)]

            # Use inference runtime for batch embedding generation
            results = await self.inference_runtime.infer_many(payloads)

            # Extract embeddings from results
            embeddings = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self._logger.error(
                        f"Embedding generation failed for text {i}: {result}"
                    )
                    # Use fallback embedding (zeros) for failed items
                    embeddings.append([0.0] * 768)  # Default embedding size
                else:
                    embeddings.append(result)

            self._logger.info(
                f"Inserting {len(texts)} articles into collection: {self.__flashpoint_id}"
            )
            self.__db.insert_documents(
                collection_name=self.__flashpoint_id,
                texts=texts,  # Stored as "documents"  use include=["documents"]
                metadatas=metadatas,  # Stored as "metadatas"  use include=["metadatas"]
                ids=ids,
                embeddings=embeddings,
            )

            self._logger.info(
                f"Vectorized {len(texts)} articles into collection: {self.__flashpoint_id}"
            )
            return self.__flashpoint_id

        except Exception as e:
            self._logger.error(f"Error vectorizing articles: {e}")
            return self.__flashpoint_id
        finally:
            # Cleanup inference runtime
            if self.inference_runtime:
                await self.inference_runtime.stop()

    async def _initialize_inference_runtime(self):
        """Initialize the inference runtime for embedding generation."""
        try:
            # Create runtime config optimized for embedding generation
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
                model_loader=self._get_embedding_model_loader, config=config
            )

            await self.inference_runtime.start()
            self._logger.info("Inference runtime initialized for embedding generation")

        except Exception as e:
            self._logger.error(f"Failed to initialize inference runtime: {e}")
            raise

    def _get_embedding_model_loader(self):
        """Model loader function for the inference runtime."""
        return ModelManager.get_embedding_model()

    def run_sync(self, feeds: List[FeedModel]) -> str:
        """
        Synchronous wrapper for backward compatibility.

        Args:
            feeds: List of feed models to vectorize

        Returns:
            str: The Chroma collection name used
        """
        import asyncio

        return asyncio.run(self.run(feeds))
