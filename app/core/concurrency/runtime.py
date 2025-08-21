"""
Unified inference runtime for MASX AI ETL.

This module provides the InferenceRuntime class that serves as the single entry point
for all inference operations, automatically selecting the optimal execution path
(GPU vs CPU) based on configuration and availability.
"""

import asyncio
import logging
import time
from typing import List, TypeVar, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass

from .device import use_gpu, get_device_config
from .cpu_executors import CPUExecutors
from .gpu_worker import GPUWorker, GPUConfig
from .model_pool import get_model_pool


def _convert_to_python_list(labels) -> List[int]:
    """
    Convert various array types to Python list for consistent serialization.

    Handles:
    - numpy.ndarray
    - cupy.ndarray
    - torch.Tensor
    - Other array-like objects
    """
    try:
        if hasattr(labels, "tolist"):
            return labels.tolist()
        elif hasattr(labels, "numpy"):
            return labels.numpy().tolist()
        elif hasattr(labels, "cpu"):
            # Handle torch.Tensor
            return labels.cpu().numpy().tolist()
        else:
            return list(labels)
    except Exception as e:
        logger.warning(f"Failed to convert labels to list: {e}, using fallback")
        return list(labels)


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class RuntimeConfig:
    """Runtime configuration parameters."""

    # GPU settings
    gpu_batch_size: int = 32
    gpu_max_delay_ms: int = 50
    gpu_queue_size: int = 1000
    gpu_timeout: float = 30.0
    gpu_use_fp16: bool = True
    gpu_enable_warmup: bool = True

    # CPU settings
    cpu_max_threads: int = 20
    cpu_max_processes: int = 4

    # Model Pool settings (for production mode)
    model_pool_max_instances: int = 2
    model_pool_enabled: bool = True

    # Debug settings
    debug_mode: Optional[bool] = None  # Auto-detect from settings if None

    # General settings
    enable_metrics: bool = True
    log_level: str = "INFO"


class InferenceRuntime:
    """
    Unified inference runtime that automatically selects optimal execution path.

    This class provides a single API for inference operations that automatically:
    - Routes to GPU worker with micro-batching when GPU is available
    - Falls back to CPU executors (thread/process pools) when GPU is not available
    - Provides consistent async interface regardless of execution path
    - Handles model loading, batching, and resource management
    """

    def __init__(
        self,
        model_loader: Optional[Callable] = None,
        config: Optional[RuntimeConfig] = None,
    ):
        """
        Initialize the inference runtime.

        Args:
            model_loader: Function that loads and returns the model (for GPU path)
            config: Runtime configuration parameters
        """
        self.model_loader = model_loader
        self.config = config or RuntimeConfig()

        # Auto-detect debug mode if not specified
        if self.config.debug_mode is None:
            try:
                from app.config import get_settings

                settings = get_settings()
                self.config.debug_mode = settings.debug
            except Exception:
                self.config.debug_mode = False  # Default to production mode

        # Device configuration
        self.device_config = get_device_config()
        self.use_gpu_flag = use_gpu()

        # Execution components
        self._gpu_worker: Optional[GPUWorker] = None
        self._cpu_executors = CPUExecutors()

        # Model pool for production mode
        self._model_pool = None
        if not self.config.debug_mode and self.config.model_pool_enabled:
            self._model_pool = get_model_pool(self.config.model_pool_max_instances)

        # Runtime state
        self._is_started = False
        self._start_time = time.time()

        # Metrics
        self._total_requests = 0
        self._total_gpu_requests = 0
        self._total_cpu_requests = 0

        mode = "Debug" if self.config.debug_mode else "Production"
        pool_info = (
            f", ModelPool={self._model_pool is not None}"
            if not self.config.debug_mode
            else ""
        )

        logger.info(
            f"InferenceRuntime initialized: Mode={mode}, GPU={self.use_gpu_flag}, "
            f"device={self.device_config.device_type}{pool_info}"
        )

    async def start(self) -> None:
        """
        Start the inference runtime.

        This method initializes the appropriate execution path based on device configuration.
        """
        if self._is_started:
            logger.warning("InferenceRuntime already started")
            return

        try:
            if self.use_gpu_flag and self.model_loader:
                await self._start_gpu_worker()
            else:
                logger.info("Using CPU execution path")

            self._is_started = True
            logger.info("InferenceRuntime started successfully")

        except Exception as e:
            logger.error(f"Failed to start InferenceRuntime: {e}")
            raise

    async def _start_gpu_worker(self):
        """Start the GPU worker."""
        if not self.model_loader:
            raise RuntimeError("Model loader required for GPU execution")

        logger.info("Starting GPU worker...")

        # Get device_id from device configuration, defaulting to 0 for GPU
        device_id = 0  # Default GPU device
        if hasattr(self.device_config, 'device_id') and self.device_config.device_id is not None:
            device_id = self.device_config.device_id

        gpu_config = GPUConfig(
            device_id=device_id,  # Pass the correct device_id
            max_batch_size=self.config.gpu_batch_size,
            max_delay_ms=self.config.gpu_max_delay_ms,
            max_queue_size=self.config.gpu_queue_size,
            timeout_seconds=self.config.gpu_timeout,
            use_fp16=self.config.gpu_use_fp16,
            enable_warmup=self.config.gpu_enable_warmup,
        )

        self._gpu_worker = GPUWorker(model_loader=self.model_loader, config=gpu_config)

        await self._gpu_worker.start()
        logger.info("GPU worker started successfully")

    async def infer(self, payload: T) -> R:
        """
        Perform single inference.

        Args:
            payload: Input data for inference

        Returns:
            Inference result
        """
        if not self._is_started:
            raise RuntimeError("InferenceRuntime not started")

        self._total_requests += 1

        try:
            if self.use_gpu_flag and self._gpu_worker:
                self._total_gpu_requests += 1
                return await self._gpu_worker.infer(payload)
            else:
                self._total_cpu_requests += 1
                # For CPU path, we need to implement the actual inference logic
                # This is a placeholder - should be customized based on actual models
                return await self._cpu_infer(payload)

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    async def infer_many(self, payloads: List[T]) -> List[R]:
        """
        Perform batch inference.

        Args:
            payloads: List of input data

        Returns:
            List of inference results
        """
        if not self._is_started:
            raise RuntimeError("InferenceRuntime not started")

        if not payloads:
            return []

        self._total_requests += len(payloads)

        try:
            if self.use_gpu_flag and self._gpu_worker:
                self._total_gpu_requests += len(payloads)
                return await self._gpu_worker.infer_many(payloads)
            else:
                self._total_cpu_requests += len(payloads)
                # For CPU path, process in parallel using appropriate executor
                try:
                    return await self._cpu_infer_many(payloads)
                except TypeError as e:
                    if "cannot pickle" in str(e):
                        logger.warning(
                            f"Pickling error in CPU batch inference, falling back to sequential: {e}"
                        )
                        # Fallback to sequential processing to avoid pickling issues
                        return await self._cpu_infer_many_sequential(payloads)
                    else:
                        raise

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise

    async def _cpu_infer(self, payload: T) -> R:
        """
        CPU-based inference with support for different payload types.

        This method handles:
        - Text summarization (with "text" field)
        - Embedding generation (with "text" + "index" fields)
        - Clustering (with "embeddings" field)
        - Debug mode: Synchronous execution (simple, reliable)
        - Production mode: Model pool with controlled concurrency
        """
        try:
            if isinstance(payload, dict):
                # Check if this is a clustering payload (has embeddings field)
                if "embeddings" in payload:
                    # This is a clustering payload - use thread pool for CPU clustering
                    logger.debug("Processing clustering payload via CPU")
                    return await self._cpu_executors.run_in_thread(
                        self._cluster_embeddings_cpu, payload
                    )

                # Check if this is a text processing payload
                elif "text" in payload:
                    # Check if this is an embedding payload (has index field)
                    if "index" in payload:
                        # This is an embedding generation payload - always use thread pool
                        return await self._cpu_executors.run_in_thread(
                            self._generate_embedding_cpu, payload
                        )
                    else:
                        # This is a summarization payload
                        serializable_payload = self._prepare_serializable_payload(
                            payload
                        )

                        if self.config.debug_mode:
                            # Debug mode: Synchronous execution (no threading/processing issues)
                            logger.debug(
                                "Using synchronous summarization for debug mode"
                            )
                            return self._summarize_text_cpu_sync(serializable_payload)
                        else:
                            # Production mode: Model pool with controlled concurrency
                            logger.debug(
                                "Using model pool summarization for production mode"
                            )
                            return await self._summarize_text_cpu_pooled(
                                serializable_payload
                            )
                else:
                    # Generic payload handling
                    logger.debug(f"CPU inference for payload: {type(payload)}")
                    return payload
            else:
                # Non-dict payload handling
                logger.debug(f"CPU inference for non-dict payload: {type(payload)}")
                return payload
        except Exception as e:
            logger.error(f"CPU inference failed: {e}")
            raise

    def _prepare_serializable_payload(self, payload: dict) -> dict:
        """
        Prepare a serializable version of the payload for thread pool execution.

        This method extracts only the necessary data from FeedModel objects to avoid
        pickling issues with non-serializable components.
        """
        try:
            feed = payload.get("feed")
            text = payload.get("text", "")
            url = payload.get("url", "")
            prompt_prefix = payload.get("prompt_prefix", "summarize: ")

            # Extract only serializable data from FeedModel
            feed_data = None
            if feed is not None:
                # Convert FeedModel to a simple dict with only serializable fields
                feed_data = {
                    "url": getattr(feed, "url", ""),
                    "title": getattr(feed, "title", ""),
                    "raw_text": getattr(feed, "raw_text", ""),
                    "summary": getattr(feed, "summary", ""),
                    "raw_text_en": getattr(feed, "raw_text_en", ""),
                    "questions": getattr(feed, "questions", []),
                    "flashpoint_id": getattr(feed, "flashpoint_id", ""),
                    "cluster_id": getattr(feed, "cluster_id", None),
                    "embedding": getattr(feed, "embedding", None),
                }

            return {
                "feed": feed_data,
                "text": text,
                "url": url,
                "prompt_prefix": prompt_prefix,
            }

        except Exception as e:
            logger.error(f"Failed to prepare serializable payload: {e}")
            # Return minimal payload if serialization fails
            return {
                "feed": None,
                "text": payload.get("text", ""),
                "url": payload.get("url", ""),
                "prompt_prefix": payload.get("prompt_prefix", "summarize: "),
            }

    def _summarize_text_cpu(self, payload: dict) -> dict:
        """
        CPU-based text summarization.

        This method runs on a separate thread to avoid blocking the event loop.
        """
        try:
            from app.singleton import ModelManager
            from app.nlp import Translator, NLPUtils

            # Extract only serializable data from payload
            feed_data = payload.get("feed")
            text = payload.get("text", "")
            url = payload.get("url", "")
            prompt_prefix = payload.get("prompt_prefix", "summarize: ")

            # Create a serializable result structure
            result = {
                "feed_data": feed_data,
                "text": text,
                "url": url,
                "translated_text": None,
                "compressed_text": None,
                "summary": None,
            }

            # Step 1: Translate non-English articles to English
            try:
                translator = Translator()
                result["translated_text"] = translator.ensure_english(text)
            except Exception as e:
                logger.error(f"Translation failed for {url}: {e}")
                result["translated_text"] = text  # Use original text as fallback

            # Step 2: Check if text fits the model, else compress using TF-IDF
            try:
                model, tokenizer, device = ModelManager.get_summarization_model()
                max_tokens = ModelManager.get_summarization_model_max_tokens()

                if not NLPUtils.text_suitable_for_model(
                    tokenizer,
                    result["translated_text"],
                    max_tokens,
                ):
                    logger.info(f"Compressing text using TF-IDF for {url}")
                    compressed_text = NLPUtils.compress_text_tfidf(
                        tokenizer, result["translated_text"], max_tokens, prompt_prefix
                    )
                    result["compressed_text"] = compressed_text
                else:
                    result["compressed_text"] = result["translated_text"]
            except Exception as e:
                logger.error(f"Text compression failed for {url}: {e}")
                result["compressed_text"] = result["translated_text"]

            # Step 3: Generate summary
            try:
                # Ensure we have valid model components
                if model is None or tokenizer is None or device is None:
                    raise RuntimeError("Model components not properly loaded")
                
                summary = NLPUtils.summarize_text(
                    model,
                    tokenizer,
                    device,
                    prompt_prefix + result["compressed_text"],
                    max_tokens,
                )
                result["summary"] = summary
                logger.info(f"Summary generated for {url}")
            except Exception as e:
                logger.error(f"Summary generation failed for {url}: {e}")
                raise

            return result

        except Exception as e:
            logger.error(f"CPU summarization failed: {e}")
            raise

    def _summarize_text_cpu_sync(self, payload: dict) -> dict:
        """
        CPU-based text summarization for debug mode (synchronous execution).

        This method runs directly in the main thread to avoid any threading issues.
        Perfect for debugging as it's simple and reliable.
        """
        try:
            from app.singleton import ModelManager
            from app.nlp import Translator, NLPUtils

            # Extract data from payload
            feed_data = payload.get("feed")
            text = payload.get("text", "")
            url = payload.get("url", "")
            prompt_prefix = payload.get("prompt_prefix", "summarize: ")

            # Create result structure
            result = {
                "feed_data": feed_data,
                "text": text,
                "url": url,
                "translated_text": None,
                "compressed_text": None,
                "summary": None,
            }

            # Step 1: Translate non-English articles to English
            try:
                translator = Translator()
                result["translated_text"] = translator.ensure_english(text)
            except Exception as e:
                logger.error(f"Translation failed for {url}: {e}")
                result["translated_text"] = text

            # Step 2: Check if text fits the model, else compress using TF-IDF
            try:
                model, tokenizer, device = ModelManager.get_summarization_model()
                max_tokens = ModelManager.get_summarization_model_max_tokens()

                if not NLPUtils.text_suitable_for_model(
                    tokenizer,
                    result["translated_text"],
                    max_tokens,
                ):
                    logger.info(f"Compressing text using TF-IDF for {url}")
                    compressed_text = NLPUtils.compress_text_tfidf(
                        tokenizer, result["translated_text"], max_tokens, prompt_prefix
                    )
                    result["compressed_text"] = compressed_text
                else:
                    result["compressed_text"] = result["translated_text"]
            except Exception as e:
                logger.error(f"Text compression failed for {url}: {e}")
                result["compressed_text"] = result["translated_text"]

            # Step 3: Generate summary (direct execution, no threading)
            try:
                summary = NLPUtils.summarize_text(
                    model,
                    tokenizer,
                    device,
                    prompt_prefix + result["compressed_text"],
                    max_tokens,
                )
                result["summary"] = summary
                logger.info(f"Summary generated for {url} (debug mode)")
            except Exception as e:
                logger.error(f"Summary generation failed for {url}: {e}")
                raise

            return result

        except Exception as e:
            logger.error(f"CPU summarization (sync) failed: {e}")
            raise

    async def _summarize_text_cpu_pooled(self, payload: dict) -> dict:
        """
        CPU-based text summarization for production mode using model pool.

        This method uses the model pool to provide controlled concurrency
        without memory explosion.
        """
        if not self._model_pool:
            # Fallback to sync if model pool not available
            logger.warning("Model pool not available, falling back to sync mode")
            return self._summarize_text_cpu_sync(payload)

        # Get model instance from pool
        model_instance = await self._model_pool.get_model(
            "summarization", lambda: self._get_summarization_model_components()
        )

        try:
            # Extract data from payload
            feed_data = payload.get("feed")
            text = payload.get("text", "")
            url = payload.get("url", "")
            prompt_prefix = payload.get("prompt_prefix", "summarize: ")

            # Create result structure
            result = {
                "feed_data": feed_data,
                "text": text,
                "url": url,
                "translated_text": None,
                "compressed_text": None,
                "summary": None,
            }

            # Step 1: Translate non-English articles to English
            try:
                from app.nlp import Translator

                translator = Translator()
                result["translated_text"] = translator.ensure_english(text)
            except Exception as e:
                logger.error(f"Translation failed for {url}: {e}")
                result["translated_text"] = text

            # Step 2: Check if text fits the model, else compress using TF-IDF
            try:
                from app.singleton import ModelManager
                from app.nlp import NLPUtils

                model = model_instance.model
                tokenizer = model_instance.tokenizer
                device = model_instance.device
                max_tokens = ModelManager.get_summarization_model_max_tokens()

                if not NLPUtils.text_suitable_for_model(
                    tokenizer,
                    result["translated_text"],
                    max_tokens,
                ):
                    logger.info(f"Compressing text using TF-IDF for {url}")
                    compressed_text = NLPUtils.compress_text_tfidf(
                        tokenizer, result["translated_text"], max_tokens, prompt_prefix
                    )
                    result["compressed_text"] = compressed_text
                else:
                    result["compressed_text"] = result["translated_text"]
            except Exception as e:
                logger.error(f"Text compression failed for {url}: {e}")
                result["compressed_text"] = result["translated_text"]

            # Step 3: Generate summary using pooled model
            try:
                summary = NLPUtils.summarize_text(
                    model,
                    tokenizer,
                    device,
                    prompt_prefix + result["compressed_text"],
                    max_tokens,
                )
                result["summary"] = summary
                logger.info(f"Summary generated for {url} (pooled mode)")
            except Exception as e:
                logger.error(f"Summary generation failed for {url}: {e}")
                raise

            return result

        except Exception as e:
            logger.error(f"CPU summarization (pooled) failed: {e}")
            raise

        finally:
            # Always return model to pool
            await self._model_pool.return_model(model_instance)

    def _get_summarization_model_components(self):
        """Helper to get summarization model components for the model pool."""
        from app.singleton import ModelManager

        return ModelManager.get_summarization_model()

    def _generate_embedding_cpu(self, payload: dict) -> list:
        """
        CPU-based embedding generation.

        This method runs on a separate thread to avoid blocking the event loop.
        """
        try:
            from app.singleton import ModelManager

            text = payload.get("text", "")
            index = payload.get("index", 0)

            # Get the embedding model
            embedding_model = ModelManager.get_embedding_model()

            # Generate embedding
            embedding = embedding_model.encode(text, convert_to_tensor=False)

            # Convert to list for serialization
            embedding_list = (
                embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            )

            logger.debug(
                f"Generated embedding for text {index} with shape: {len(embedding_list)}"
            )

            return embedding_list

        except Exception as e:
            logger.error(f"CPU embedding generation failed: {e}")
            raise

    def _cluster_embeddings_cpu(self, payload: dict) -> list:
        """
        CPU-based clustering for HDBSCAN/KMeans.

        This method runs on a separate thread to avoid blocking the event loop.
        """
        try:
            import numpy as np
            import hdbscan
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize

            # Extract clustering parameters
            embeddings = np.array(payload.get("embeddings", []))
            use_umap = payload.get("use_umap", True)
            umap_n_components = payload.get("umap_n_components", 20)
            umap_n_neighbors = payload.get("umap_n_neighbors", 15)
            umap_min_dist = payload.get("umap_min_dist", 0.1)
            min_cluster_size = payload.get("min_cluster_size", 3)
            min_samples = payload.get("min_samples", 2)
            metric = payload.get("metric", "euclidean")
            cluster_selection_method = payload.get("cluster_selection_method", "eom")
            cluster_selection_epsilon = payload.get("cluster_selection_epsilon", 0.05)
            allow_single_cluster = payload.get("allow_single_cluster", False)
            random_state = payload.get("random_state", 42)

            if len(embeddings) == 0:
                logger.warning("Empty embeddings array provided for clustering")
                return []

            # 1) Normalize to unit sphere for cosine geometry
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if metric.lower() in ("cosine", "euclidean"):
                embeddings = normalize(embeddings, norm="l2", axis=1, copy=False)

            # 2) Optional UMAP reduction (CPU only)
            if use_umap:
                try:
                    import umap

                    reducer = umap.UMAP(
                        n_components=umap_n_components,
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist,
                        metric="cosine",  # we normalized to unit sphere
                        random_state=random_state,
                    )
                    embeddings = reducer.fit_transform(embeddings)
                    logger.debug(f"UMAP reduction applied: {embeddings.shape}")
                except ImportError:
                    logger.warning(
                        "UMAP not available, skipping dimensionality reduction"
                    )
                except Exception as e:
                    logger.warning(
                        f"UMAP reduction failed: {e}, using original embeddings"
                    )

            # 3) Perform clustering
            if metric.lower() in ("cosine", "euclidean"):
                # Use euclidean because on the unit sphere it ≈ cosine
                cluster_metric = "euclidean"
            else:
                cluster_metric = metric

            # Adjust cluster selection method for HDBSCAN
            if cluster_selection_method == "eom":
                csm = "leaf"
            else:
                csm = cluster_selection_method

            # Use HDBSCAN for clustering
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=cluster_metric,
                cluster_selection_method=csm,
                cluster_selection_epsilon=cluster_selection_epsilon,
                allow_single_cluster=allow_single_cluster,
                prediction_data=True,
                approx_min_span_tree=True,
            )

            labels = model.fit_predict(embeddings).astype(np.int32)
            logger.info(
                f"CPU clustering completed: {len(labels)} labels, {len(set(labels) - {-1})} clusters"
            )

            # Convert numpy array to Python list for serialization
            return _convert_to_python_list(labels)

        except Exception as e:
            logger.error(f"CPU clustering failed: {e}")
            raise

    async def _cpu_infer_many(self, payloads: List[T]) -> List[R]:
        """
        CPU-based batch inference with proper error handling.

        This method processes payloads in parallel using thread pool to avoid pickling issues.
        """
        logger.debug(f"CPU batch inference for {len(payloads)} payloads")

        # Process in parallel using thread pool (not process pool to avoid pickling issues)
        tasks = [self._cpu_infer(payload) for payload in payloads]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _cpu_infer_many_sequential(self, payloads: List[T]) -> List[R]:
        """
        Sequential CPU-based batch inference as fallback.

        This method processes payloads sequentially to avoid any concurrency issues.
        """
        logger.debug(f"Sequential CPU batch inference for {len(payloads)} payloads")

        results = []
        for payload in payloads:
            try:
                result = await self._cpu_infer(payload)
                results.append(result)
            except Exception as e:
                logger.error(f"Sequential inference failed for payload: {e}")
                results.append(e)  # Return exception as result

        return results

    async def stop(self) -> None:
        """
        Stop the inference runtime gracefully.

        This method ensures proper cleanup of all resources.
        """
        if not self._is_started:
            return

        logger.info("Stopping InferenceRuntime...")

        try:
            # Stop GPU worker if running
            if self._gpu_worker:
                await self._gpu_worker.stop()
                self._gpu_worker = None

            # Shutdown CPU executors
            self._cpu_executors.shutdown(wait=True)

            # Clean up model pool if in production mode
            if self._model_pool:
                await self._model_pool.cleanup()
                self._model_pool = None

            self._is_started = False
            logger.info("InferenceRuntime stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping InferenceRuntime: {e}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get runtime metrics."""
        uptime = time.time() - self._start_time

        metrics = {
            "runtime_uptime_seconds": uptime,
            "total_requests": self._total_requests,
            "total_gpu_requests": self._total_gpu_requests,
            "total_cpu_requests": self._total_cpu_requests,
            "requests_per_second": self._total_requests / uptime if uptime > 0 else 0,
            "is_started": self._is_started,
            "use_gpu": self.use_gpu_flag,
            "device_type": self.device_config.device_type,
        }

        # Add GPU worker metrics if available
        if self._gpu_worker:
            metrics["gpu_worker"] = self._gpu_worker.get_metrics()

        # Add CPU executor metrics
        metrics["cpu_executors"] = {
            "max_threads": self._cpu_executors.max_threads,
            "max_processes": self._cpu_executors.max_processes,
            "is_shutdown": self._cpu_executors.is_shutdown(),
        }

        # Add model pool metrics if available
        if self._model_pool:
            try:
                import asyncio

                # Create a task to get pool stats (since get_metrics is sync)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in sync method, so just note that pool is available
                    metrics["model_pool"] = {
                        "enabled": True,
                        "max_instances_per_type": self.config.model_pool_max_instances,
                        "stats": "available_via_async_call",
                    }
                else:
                    pool_stats = loop.run_until_complete(
                        self._model_pool.get_pool_stats()
                    )
                    metrics["model_pool"] = {
                        "enabled": True,
                        "max_instances_per_type": self.config.model_pool_max_instances,
                        "stats": pool_stats,
                    }
            except Exception:
                metrics["model_pool"] = {
                    "enabled": True,
                    "max_instances_per_type": self.config.model_pool_max_instances,
                    "stats": "unavailable",
                }
        else:
            metrics["model_pool"] = {
                "enabled": False,
                "reason": "debug_mode" if self.config.debug_mode else "disabled",
            }

        return metrics

    @property
    def is_started(self) -> bool:
        """Check if runtime is started."""
        return self._is_started

    @property
    def queue_depth(self) -> int:
        """Get current queue depth (GPU only)."""
        if self._gpu_worker:
            return self._gpu_worker.queue_depth
        return 0

    @property
    def execution_path(self) -> str:
        """Get current execution path."""
        if self.use_gpu_flag and self._gpu_worker:
            return "gpu"
        return "cpu"


# Convenience function for easy access
async def create_runtime(
    model_loader: Optional[Callable] = None, config: Optional[RuntimeConfig] = None
) -> InferenceRuntime:
    """
    Create and start an inference runtime.

    Args:
        model_loader: Function that loads and returns the model
        config: Runtime configuration parameters

    Returns:
        Started InferenceRuntime instance
    """
    runtime = InferenceRuntime(model_loader=model_loader, config=config)
    await runtime.start()
    return runtime
