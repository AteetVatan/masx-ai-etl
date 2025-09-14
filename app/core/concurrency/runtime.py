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
#from .cpu_executors import CPUExecutors
#from .gpu_worker import GPUWorker, GPUConfig
from .model_pool import get_model_pool
from app.config import get_settings, get_service_logger
from app.singleton import ModelManager
from app.core.models import AbstractModel

# from app.nlp import Translator, NLPUtils


def get_summarizer_utils():
    """Lazy import to avoid circular dependency."""
    from app.etl.tasks import SummarizerUtils

    return SummarizerUtils


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
        print(
            f"runtime.py:Failed to convert labels to list: {e}, using fallback"
        )
        return list(labels)




T = TypeVar("T")
R = TypeVar("R")


@dataclass
class RuntimeConfig:
    """Runtime configuration parameters."""
    settings = get_settings()
    # Model Pool settings (for production mode)
    model_pool_max_instances: int = settings.model_pool_max_instances
    # Debug settings
    debug_mode: Optional[bool] = settings.debug
    # General settings
    enable_metrics: bool = settings.enable_metrics
    log_level: str = settings.log_level


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
        model_manager_loader: Optional[Callable] = None,
        config: Optional[RuntimeConfig] = None,
    ):
        """
        Initialize the inference runtime.

        Args:
            model_pool_loader: Function that loads and returns the model pool (for GPU path)
            config: Runtime configuration parameters
        """
        self.model_manager_loader = model_manager_loader
        self.model_manager: AbstractModel = None
        self.config = config or RuntimeConfig()
        self.logger = get_service_logger("InferenceRuntime")

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
        #self._gpu_worker: Optional[GPUWorker] = None
        #self._cpu_executors = CPUExecutors()

        # Model pool for production mode
        # self._model_pool = None
        # if not self.config.debug_mode and self.config.model_pool_enabled:
        #     self._model_pool = get_model_pool(self.config.model_pool_max_instances)

        # Runtime state
        self._is_started = False
        self._start_time = time.time()

        # Metrics
        self._total_requests = 0
        self._total_gpu_requests = 0
        self._total_cpu_requests = 0
        

        mode = "Debug" if self.config.debug_mode else "Production"
        # pool_info = (
        #     f", ModelPool={self._model_pool is not None}"
        #     if not self.config.debug_mode
        #     else ""
        # )

        # self.logger.info(
        #     f"runtime.py:InferenceRuntime initialized: Mode={mode}, GPU={self.use_gpu_flag}, "
        #     f"device={self.device_config.device_type}{pool_info}"
        # )

    async def start(self) -> None:
        """
        Start the inference runtime.

        This method initializes the appropriate execution path based on device configuration.
        """
        if self._is_started:
            self.logger.warning("runtime.py:InferenceRuntime already started")
            return

        try:
            # if self.use_gpu_flag and self.model_pool_loader:
            #     await self._start_gpu_worker()
            # else:
            #     self.logger.info("runtime.py:Using CPU execution path")
                
            self.model_manager = self.model_manager_loader()   
            self.model_manager.initialize()
                

            self._is_started = True
            self.logger.info("runtime.py:InferenceRuntime started successfully")

        except Exception as e:
            self.logger.error(f"runtime.py:Failed to start InferenceRuntime: {e}")
            raise

    # async def _start_gpu_worker(self):
    #     """Start the GPU worker."""
    #     if not self.model_pool_loader:
    #         raise RuntimeError("Model loader required for GPU execution")

    #     self.logger.info("runtime.py:Starting GPU worker...")

    #     # Get device_id from device configuration, defaulting to 0 for GPU
    #     device_id = 0  # Default GPU device
    #     if (
    #         hasattr(self.device_config, "device_id")
    #         and self.device_config.device_id is not None
    #     ):
    #         device_id = self.device_config.device_id

    #     gpu_config = GPUConfig(
    #         device_id=device_id,  # Pass the correct device_id
    #         max_batch_size=self.config.gpu_batch_size,
    #         max_delay_ms=self.config.gpu_max_delay_ms,
    #         max_queue_size=self.config.gpu_queue_size,
    #         timeout_seconds=self.config.gpu_timeout,
    #         use_fp16=self.config.gpu_use_fp16,
    #         enable_warmup=self.config.gpu_enable_warmup,
    #     )

    #     self._gpu_worker = GPUWorker(model_loader=self.model_pool_loader, config=gpu_config)

    #     await self._gpu_worker.start()
    #     self.logger.info("runtime.py:GPU worker started successfully")

    # async def infer(self, payload: T) -> R:
    #     """
    #     Perform single inference.

    #     Args:
    #         payload: Input data for inference

    #     Returns:
    #         Inference result
    #     """
    #     if not self._is_started:
    #         raise RuntimeError("InferenceRuntime not started")

    #     self._total_requests += 1

    #     try:
    #         if self.use_gpu_flag and self._gpu_worker:
    #             self._total_gpu_requests += 1
    #             return await self._gpu_worker.infer(payload)
    #         else:
    #             self._total_cpu_requests += 1
    #             # For CPU path, we need to implement the actual inference logic
    #             # This is a placeholder - should be customized based on actual models
    #             return await self._cpu_infer(payload)

    #     except Exception as e:
    #         self.logger.error(f"runtime.py:Inference failed: {e}")
    #         raise

    # async def infer_many(self, payloads: List[T]) -> List[R]:
    #     """
    #     Perform batch inference.

    #     Args:
    #         payloads: List of input data

    #     Returns:
    #         List of inference results
    #     """
    #     if not self._is_started:
    #         raise RuntimeError("InferenceRuntime not started")

    #     if not payloads:
    #         return []

    #     self._total_requests += len(payloads)

    #     try:
    #         if self.use_gpu_flag and self._gpu_worker:
    #             self.logger.info(f"runtime.py:infer_many using GPU batch inference")
    #             self._total_gpu_requests += len(payloads)
    #             return await self._gpu_worker.infer_many(payloads)
    #         else:
    #             self.logger.info(f"runtime.py:infer_many using CPU batch inference")
    #             self._total_cpu_requests += len(payloads)
    #             # For CPU path, process in parallel using appropriate executor
    #             try:
    #                 return await self._cpu_infer_many(payloads)
    #             except TypeError as e:
    #                 if "cannot pickle" in str(e):
    #                     self.logger.warning(
    #                         f"runtime.py:Pickling error in CPU batch inference, falling back to sequential: {e}"
    #                     )
    #                     # Fallback to sequential processing to avoid pickling issues
    #                     return await self._cpu_infer_many_sequential(payloads)
    #                 else:
    #                     raise

    #     except Exception as e:
    #         self.logger.error(f"runtime.py:Batch inference failed: {e}")
    #         raise

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
        pass
        # try:
        #     if isinstance(payload, dict):
        #         # Check if this is a clustering payload (has embeddings field)
        #         if "embeddings" in payload:
        #             # This is a clustering payload - use thread pool for CPU clustering
        #             self.logger.debug("runtime.py:Processing clustering payload via CPU")
        #             return await self._cpu_executors.run_in_thread(
        #                 self._cluster_embeddings_cpu, payload
        #             )

        #         # Check if this is a text processing payload
        #         elif "text" in payload:
        #             # Check if this is an embedding payload (has index field)
        #             if "index" in payload:
        #                 # This is an embedding generation payload - always use thread pool
        #                 return await self._cpu_executors.run_in_thread(
        #                     self._generate_embedding_cpu, payload
        #                 )
        #             else:
        #                 # This is a summarization payload
        #                 serializable_payload = self._prepare_serializable_payload(
        #                     payload
        #                 )

        #                 if self.config.debug_mode:
        #                     # Debug mode: Synchronous execution (no threading/processing issues)
        #                     self.logger.debug(
        #                         "Using synchronous summarization for debug mode"
        #                     )
        #                     return self._summarize_text_cpu_sync(serializable_payload)
        #                 else:
        #                     # Production mode: Model pool with controlled concurrency
        #                     self.logger.debug(
        #                         "Using model pool summarization for production mode"
        #                     )
        #                     return await self._summarize_text_cpu_pooled(
        #                         serializable_payload
        #                     )
        #         else:
        #             # Generic payload handling
        #             self.logger.debug(
        #                 f"runtime.py:CPU inference for payload: {type(payload)}"
        #             )
        #             return payload
        #     else:
        #         # Non-dict payload handling
        #         self.logger.debug(
        #             f"runtime.py:CPU inference for non-dict payload: {type(payload)}"
        #         )
        #         return payload
        # except Exception as e:
        #     self.logger.error(f"runtime.py:CPU inference failed: {e}")
        #     raise

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
            self.logger.error(f"runtime.py:Failed to prepare serializable payload: {e}")
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
        model, tokenizer, device = ModelManager.get_summarization_model()
        max_tokens = ModelManager.get_summarization_model_max_tokens()
        summarizer_utils = get_summarizer_utils()
        result = summarizer_utils._summarizer(
            payload, model, tokenizer, device, max_tokens
        )
        return result

    def _summarize_text_cpu_sync(self, payload: dict) -> dict:
        """
        CPU-based text summarization for debug mode (synchronous execution).

        This method runs directly in the main thread to avoid any threading issues.
        Perfect for debugging as it's simple and reliable.
        """
        try:
            model, tokenizer, device = ModelManager.get_summarization_model()
            max_tokens = ModelManager.get_summarization_model_max_tokens()
            summarizer_utils = get_summarizer_utils()
            result = summarizer_utils._summarizer(
                payload, model, tokenizer, device, max_tokens
            )
            return result
        except Exception as e:
            self.logger.error(f"runtime.py:CPU summarization (sync) failed: {e}")
            raise

    async def _summarize_text_cpu_pooled(self, payload: dict) -> dict:
        """
        CPU-based text summarization (pooled) — UPDATED to implement:
        Preprocess → Adaptive Compress → Map-Reduce (overlap) → Merge & Polish → Quality Gates
        """
        # logger = getattr(self, "logger", None) or __import__("logging").getLogger(__name__)
        if not self._model_pool:
            self.logger.warning(
                "runtime.py:Model pool not available, falling back to sync mode"
            )
            return self._summarize_text_cpu_sync(payload)

        model_instance = await self._model_pool.get_model(
            "summarization", lambda: self._get_summarization_model_components()
        )

        try:
            # -------- Load model/tokenizer --------
            model = model_instance.model
            tokenizer = model_instance.tokenizer
            device = model_instance.device
            max_tokens = (
                ModelManager.get_summarization_model_max_tokens()
            )  # encoder limit (e.g., 1024)
            summarizer_utils = get_summarizer_utils()
            result = summarizer_utils._summarizer(
                payload, model, tokenizer, device, max_tokens
            )
            return result
        except Exception as e:
            self.logger.error(f"runtime.py:CPU summarization (pooled) failed: {e}")
            raise
        finally:
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

            self.logger.debug(
                f"runtime.py:Generated embedding for text {index} with shape: {len(embedding_list)}"
            )

            return embedding_list

        except Exception as e:
            self.logger.error(f"runtime.py:CPU embedding generation failed: {e}")
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
                self.logger.warning(
                    "runtime.py:Empty embeddings array provided for clustering"
                )
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
                    self.logger.debug(
                        f"runtime.py:UMAP reduction applied: {embeddings.shape}"
                    )
                except ImportError:
                    self.logger.warning(
                        "runtime.py:UMAP not available, skipping dimensionality reduction"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"runtime.py:UMAP reduction failed: {e}, using original embeddings"
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
            self.logger.info(
                f"runtime.py:CPU clustering completed: {len(labels)} labels, {len(set(labels) - {-1})} clusters"
            )

            # Convert numpy array to Python list for serialization
            return _convert_to_python_list(labels)

        except Exception as e:
            self.logger.error(f"runtime.py:CPU clustering failed: {e}")
            raise

    async def _cpu_infer_many(self, payloads: List[T]) -> List[R]:
        """
        CPU-based batch inference with proper error handling.

        This method processes payloads in parallel using thread pool to avoid pickling issues.
        """
        self.logger.debug(f"runtime.py:CPU batch inference for {len(payloads)} payloads")

        # Process in parallel using thread pool (not process pool to avoid pickling issues)
        tasks = [self._cpu_infer(payload) for payload in payloads]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _cpu_infer_many_sequential(self, payloads: List[T]) -> List[R]:
        """
        Sequential CPU-based batch inference as fallback.

        This method processes payloads sequentially to avoid any concurrency issues.
        """
        self.logger.debug(
            f"runtime.py:Sequential CPU batch inference for {len(payloads)} payloads"
        )

        results = []
        for payload in payloads:
            try:
                result = await self._cpu_infer(payload)
                results.append(result)
            except Exception as e:
                self.logger.error(f"runtime.py:Sequential inference failed for payload: {e}")
                results.append(e)  # Return exception as result

        return results
    
    # async def stop(self) -> None:
    #     """
    #     Stop the inference runtime gracefully.
    #     """
    #     if self.model_manager:
    #         await self.model_manager.release_all_instances()       
       
       
            

    # async def stop(self) -> None:
    #     """
    #     Stop the inference runtime gracefully.

    #     This method ensures proper cleanup of all resources.
    #     """
    #     if not self._is_started:
    #         return

    #     self.logger.info("runtime.py:Stopping InferenceRuntime...")

    #     try:
    #         # Stop GPU worker if running
    #         if self._gpu_worker:
    #             await self._gpu_worker.stop()
    #             self._gpu_worker = None

    #         # Shutdown CPU executors
    #         self._cpu_executors.shutdown(wait=True)


    #         self._is_started = False
    #         self.logger.info("runtime.py:InferenceRuntime stopped successfully")

    #     except Exception as e:
    #         self.logger.error(f"runtime.py:Error stopping InferenceRuntime: {e}")
    #         raise

    # def get_metrics(self) -> Dict[str, Any]:
    #     """Get runtime metrics."""
    #     uptime = time.time() - self._start_time

    #     metrics = {
    #         "runtime_uptime_seconds": uptime,
    #         "total_requests": self._total_requests,
    #         "total_gpu_requests": self._total_gpu_requests,
    #         "total_cpu_requests": self._total_cpu_requests,
    #         "requests_per_second": self._total_requests / uptime if uptime > 0 else 0,
    #         "is_started": self._is_started,
    #         "use_gpu": self.use_gpu_flag,
    #         "device_type": self.device_config.device_type,
    #     }

    #     # Add GPU worker metrics if available
    #     if self._gpu_worker:
    #         metrics["gpu_worker"] = self._gpu_worker.get_metrics()

    #     # Add CPU executor metrics
    #     metrics["cpu_executors"] = {
    #         "max_threads": self._cpu_executors.max_threads,
    #         "max_processes": self._cpu_executors.max_processes,
    #         "is_shutdown": self._cpu_executors.is_shutdown(),
    #     }

    #     # Add model pool metrics if available
    #     if self._model_pool:
    #         try:
    #             import asyncio

    #             # Create a task to get pool stats (since get_metrics is sync)
    #             loop = asyncio.get_event_loop()
    #             if loop.is_running():
    #                 # Can't await in sync method, so just note that pool is available
    #                 metrics["model_pool"] = {
    #                     "enabled": True,
    #                     "max_instances_per_type": self.config.model_pool_max_instances,
    #                     "stats": "available_via_async_call",
    #                 }
    #             else:
    #                 pool_stats = loop.run_until_complete(
    #                     self._model_pool.get_pool_stats()
    #                 )
    #                 metrics["model_pool"] = {
    #                     "enabled": True,
    #                     "max_instances_per_type": self.config.model_pool_max_instances,
    #                     "stats": pool_stats,
    #                 }
    #         except Exception:
    #             metrics["model_pool"] = {
    #                 "enabled": True,
    #                 "max_instances_per_type": self.config.model_pool_max_instances,
    #                 "stats": "unavailable",
    #             }
    #     else:
    #         metrics["model_pool"] = {
    #             "enabled": False,
    #             "reason": "debug_mode" if self.config.debug_mode else "disabled",
    #         }

    #     return metrics

    # @property
    # def is_started(self) -> bool:
    #     """Check if runtime is started."""
    #     return self._is_started

    # @property
    # def queue_depth(self) -> int:
    #     """Get current queue depth (GPU only)."""
    #     if self._gpu_worker:
    #         return self._gpu_worker.queue_depth
    #     return 0

    # @property
    # def execution_path(self) -> str:
    #     """Get current execution path."""
    #     if self.use_gpu_flag and self._gpu_worker:
    #         return "gpu"
    #     return "cpu"


# # Convenience function for easy access
# async def create_runtime(
#     model_manager_loader: Optional[Callable] = None, config: Optional[RuntimeConfig] = None
# ) -> InferenceRuntime:
#     """
#     Create and start an inference runtime.

#     Args:
#         model_loader: Function that loads and returns the model
#         config: Runtime configuration parameters

#     Returns:
#         Started InferenceRuntime instance
#     """
#     runtime = InferenceRuntime(model_manager_loader=model_manager_loader, config=config)
#     await runtime.start()
#     return runtime
