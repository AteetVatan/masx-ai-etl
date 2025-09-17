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
from app.nlp import VectorDBManager
from app.etl_data.etl_models.feed_model import FeedModel
from app.config import get_service_logger, get_settings
from app.core.concurrency import InferenceRuntime, RuntimeConfig
from app.core.model import VectorEmbeddingModelManager
from app.core.concurrency import CPUExecutors
from app.enumeration import WorkloadEnums


class VectorizeTask:
    """ETL Task: Vectorize Feed objects and store them using VectorDBManager."""

    def __init__(self, flashpoint_id: str):
        self.__flashpoint_id = flashpoint_id
        self.__db = VectorDBManager()
        self._logger = get_service_logger("VectorizeTask")
        self.settings = get_settings()
        # Initialize inference runtime for embedding generation
        self.inference_runtime: Optional[InferenceRuntime] = None
        self.cpu_executors = CPUExecutors(workload=WorkloadEnums.CPU)

    # gettters
    def get_flashpoint_id(self) -> str:
        return self.__flashpoint_id

    def get_db(self) -> VectorDBManager:
        return self.__db

    async def run(self, feeds: List[FeedModel]) -> str:
        """
        Vectorize and store Feeds in Chroma vector DB using async inference runtime.

        Uses `.summary` if available, otherwise falls back to `.raw_text`.

        Returns:
            str: The Chroma collection name used
        """
        try:
            self._logger.info(f"VectorizeTask: vectorizing {len(feeds)} feeds")

            # Initialize inference runtime if not already done
            if not self.inference_runtime:
                self._logger.info("VectorizeTask: initializing inference runtime")
                await self._initialize_inference_runtime()

            embedding_manager: VectorEmbeddingModelManager = (
                self.inference_runtime.model_manager
            )
            batch_size = embedding_manager.pool_size
            vectorized_feeds: List[FeedModel] = []

            # Process feeds in batches
            for i in range(0, len(feeds), batch_size):
                batch = feeds[i : i + batch_size]
                self._logger.info(
                    f"VectorizeTask: processing batch of {len(batch)} feeds"
                )
                batch_feeds = await self._process_batch(
                    batch
                )  # parallel execution of the batch
                vectorized_feeds.extend(batch_feeds)

            # Store all vectorized feeds in the database
            if vectorized_feeds:
                await self._store_vectorized_feeds(vectorized_feeds)

            return self.__flashpoint_id

        except Exception as e:
            self._logger.error(f"VectorizeTask: error vectorizing feeds: {e}")
            return self.__flashpoint_id
        finally:
            # Final cleanup -- remove all the models from the pool
            if self.inference_runtime:
                self.inference_runtime.model_manager.cleanup()
            if self.cpu_executors:
                self.cpu_executors.shutdown(wait=True)

    async def _initialize_inference_runtime(self):
        """Initialize the inference runtime for embedding generation."""
        try:
            # Create runtime config optimized for embedding generation
            config = RuntimeConfig()

            # Create and start inference runtime
            self.inference_runtime = InferenceRuntime(
                model_manager_loader=self._get_embedding_model_manager, config=config
            )

            await self.inference_runtime.start()
            self._logger.info(
                "VectorizeTask: Inference runtime initialized for embedding generation"
            )

        except Exception as e:
            self._logger.error(
                f"VectorizeTask: Failed to initialize inference runtime: {e}"
            )
            raise

    def _get_embedding_model_manager(self):
        """Model loader function for the inference runtime."""
        return VectorEmbeddingModelManager(self.settings)

    async def _process_batch(self, feeds: List[FeedModel]) -> List[FeedModel]:
        """Process a batch of feeds using the inference runtime with CPU/GPU micro-batching."""
        try:
            import asyncio

            embedding_manager: VectorEmbeddingModelManager = (
                self.inference_runtime.model_manager
            )

            async def run(feed: FeedModel) -> FeedModel | None:
                # each task acquires its own instance
                async with embedding_manager.acquire(
                    destroy_after_use=False
                ) as instance:
                    try:
                        # Generate embedding for the feed text
                        text = feed.processed_text.strip()

                        # Encode text into tokens
                        encoded = instance.tokenizer.encode(
                            text, add_special_tokens=True
                        )

                        print("Number of tokens:", len(encoded))
                        print(
                            "Max tokens allowed:", instance.tokenizer.model_max_length
                        )

                        if not text:
                            return None

                        embedding = await self.cpu_executors.run_in_thread(
                            embedding_manager.encode_text, text, instance
                        )

                        if embedding is None:
                            raise Exception(
                                "VectorizeTask: Error generating embedding for feed"
                            )

                        # Store the embedding in the feed object for later use
                        feed.embedding = embedding
                        return feed
                    except Exception as e:
                        self._logger.error(f"VectorizeTask: Error processing feed: {e}")
                        return None

            # schedule all feeds in parallel
            tasks = [run(feed) for feed in feeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # filter out None values and exceptions
            processed_feeds = [feed for feed in results if isinstance(feed, FeedModel)]
            return processed_feeds

        except Exception as e:
            self._logger.error(f"VectorizeTask: Batch processing failed: {e}")
            return []

    async def _store_vectorized_feeds(self, feeds: List[FeedModel]) -> None:
        """Store vectorized feeds in the vector database."""
        try:
            texts, metadatas, ids, embeddings = [], [], [], []

            for feed in feeds:
                if (
                    not hasattr(feed, "embedding")
                    or feed.embedding is None
                    or len(feed.embedding) == 0
                ):
                    continue

                text = feed.summary.strip() if feed.summary else feed.raw_text.strip()
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
                ids.append(str(uuid4()))  # each vector gets a unique ID
                embeddings.append(feed.embedding)

            if not texts:
                self._logger.info("No valid feeds to store in vector database.")
                return

            self._logger.info(
                f"VectorizeTask: Inserting {len(texts)} feeds into collection: {self.__flashpoint_id}"
            )

            self.__db.insert_documents(
                collection_name=self.__flashpoint_id,
                texts=texts,  # Stored as "documents"  use include=["documents"]
                metadatas=metadatas,  # Stored as "metadatas"  use include=["metadatas"]
                ids=ids,
                embeddings=embeddings,
            )

            self._logger.info(
                f"VectorizeTask: Vectorized {len(texts)} feeds into collection: {self.__flashpoint_id}"
            )

        except Exception as e:
            self._logger.error(f"VectorizeTask: Error storing vectorized feeds: {e}")
            raise

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
