"""
GPU worker for optimized inference with micro-batching.

This module provides a dedicated GPU worker that handles model loading, inference,
and micro-batching to maximize GPU utilization while maintaining low latency.
"""

import asyncio
import logging
import time
from typing import List, TypeVar, Optional, Dict, Any, Union
from dataclasses import dataclass
import threading

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .device import use_gpu, get_device_config
from .batcher import MicroBatcher

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class GPUConfig:
    """GPU configuration parameters."""

    device_id: int = 0
    max_batch_size: int = 32
    max_delay_ms: int = 50
    max_queue_size: int = 1000
    timeout_seconds: float = 7200.0  # 2 hours for ETL processes
    use_fp16: bool = True
    enable_warmup: bool = True
    pinned_memory: bool = True
    non_blocking: bool = True


class GPUWorker:
    """
    Dedicated GPU worker for inference with micro-batching.

    This class provides:
    - Single GPU model instance per process
    - Micro-batching for optimal throughput
    - Automatic device management and error handling
    - Metrics and monitoring
    """

    def __init__(self, model_loader: callable, config: Optional[GPUConfig] = None):
        """
        Initialize GPU worker.

        Args:
            model_loader: Function that loads and returns the model
            config: GPU configuration parameters
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        if not use_gpu():
            raise RuntimeError("GPU not available or disabled")

        self.model_loader = model_loader
        self.config = config or GPUConfig()

        # Validate device_id
        if not isinstance(self.config.device_id, int) or self.config.device_id < 0:
            raise ValueError(
                f"Invalid device_id: {self.config.device_id}. Must be a non-negative integer."
            )

        # Model state
        self._model = None
        self._model_lock = threading.Lock()
        self._is_initialized = False

        # Micro-batcher
        self._batcher = MicroBatcher(
            batch_processor=self._process_batch,
            max_batch_size=self.config.max_batch_size,
            max_delay_ms=self.config.max_delay_ms,
            max_queue_size=self.config.max_queue_size,
            timeout_seconds=self.config.timeout_seconds,
        )

        # Metrics
        self._total_inferences = 0
        self._total_batch_time = 0.0
        self._start_time = time.time()

        logger.info(
            f"gpu_worker.py:GPUWorker initialized on device {self.config.device_id}"
        )

    async def start(self):
        """Start the GPU worker and load model."""
        try:
            await self._load_model()
            if self.config.enable_warmup:
                await self._warmup_model()

            self._is_initialized = True
            logger.info("gpu_worker.py:GPUWorker started successfully")

        except Exception as e:
            logger.error(f"gpu_worker.py:Failed to start GPUWorker: {e}")
            raise

    async def _load_model(self):
        """Load the model on GPU."""
        with self._model_lock:
            if self._model is not None:
                return

            logger.info("gpu_worker.py:Loading model on GPU...")

            try:
                # Load model using the provided loader
                self._model = self.model_loader()

                # Validate device_id before creating device string
                if (
                    not isinstance(self.config.device_id, int)
                    or self.config.device_id < 0
                ):
                    raise ValueError(f"Invalid device_id: {self.config.device_id}")

                # Move to GPU
                device = torch.device(f"cuda:{self.config.device_id}")
                self._model = self._model.to(device)

                # Set to evaluation mode
                if hasattr(self._model, "eval"):
                    self._model.eval()

                # Enable FP16 if requested and supported
                if self.config.use_fp16 and device.type == "cuda":
                    try:
                        self._model = self._model.half()
                        logger.info("gpu_worker.py:Model converted to FP16")
                    except Exception as e:
                        logger.warning(f"gpu_worker.py:FP16 conversion failed: {e}")

                logger.info(f"gpu_worker.py:Model loaded successfully on {device}")

            except Exception as e:
                logger.error(f"gpu_worker.py:Model loading failed: {e}")
                raise

    async def _warmup_model(self):
        """Warm up the model with dummy inference."""
        if not self._model:
            return

        logger.info("gpu_worker.py:Warming up model...")

        try:
            # Create dummy input based on model type
            dummy_input = self._create_dummy_input()

            # Warmup inference
            with torch.no_grad():
                if hasattr(self._model, "forward"):
                    _ = self._model(dummy_input)
                elif callable(self._model):
                    _ = self._model(dummy_input)

            logger.info("gpu_worker.py:Model warmup completed")

        except Exception as e:
            logger.warning(f"gpu_worker.py:Model warmup failed: {e}")

    def _create_dummy_input(self):
        """Create dummy input for model warmup."""
        # This is a placeholder - should be customized based on actual model
        try:
            # Validate device_id before creating device strings
            if not isinstance(self.config.device_id, int) or self.config.device_id < 0:
                raise ValueError(f"Invalid device_id: {self.config.device_id}")

            # Try to create input based on model's expected input shape
            if hasattr(self._model, "config"):
                # For transformers models
                if hasattr(self._model.config, "max_position_embeddings"):
                    seq_len = min(self._model.config.max_position_embeddings, 512)
                    return torch.randint(0, 1000, (1, seq_len)).to(
                        torch.device(f"cuda:{self.config.device_id}")
                    )

            # Generic fallback
            return torch.randn(1, 128).to(torch.device(f"cuda:{self.config.device_id}"))

        except Exception:
            # Final fallback
            return torch.randn(1, 128).to(torch.device(f"cuda:{self.config.device_id}"))

    async def infer(self, payload: T) -> R:
        """
        Submit a single inference request.

        Args:
            payload: Input data for inference

        Returns:
            Inference result
        """
        if not self._is_initialized:
            raise RuntimeError("GPUWorker not initialized")

        return await self._batcher.submit(payload)

    async def infer_many(self, payloads: List[T]) -> List[R]:
        """
        Submit multiple inference requests.

        Args:
            payloads: List of input data

        Returns:
            List of inference results
        """
        if not self._is_initialized:
            raise RuntimeError("GPUWorker not initialized")

        tasks = [self.infer(payload) for payload in payloads]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_batch(self, payloads: List[T]) -> List[R]:
        """
        Process a batch of inference requests.

        Args:
            payloads: Batch of input data

        Returns:
            Batch of inference results
        """
        logger.info(f"gpu_worker.py:GPUWorker:_process_batch batch inference")
        if not self._model:
            raise RuntimeError("Model not loaded")

        batch_size = len(payloads)
        start_time = time.time()

        try:
            # Prepare batch input
            batch_input = await self._prepare_batch_input(payloads)

            # Check if this is a summarization batch
            if isinstance(batch_input, dict) and "model" in batch_input:
                # This is a summarization batch - the model is in the batch_input
                # For summarization, we pass the entire batch_input to process_batch_output
                # SummarizerUtils._summarizer will handle all the processing
                batch_output = batch_input
            else:
                # Generic model inference
                with torch.no_grad():
                    if hasattr(self._model, "forward"):
                        batch_output = self._model(batch_input)
                    elif callable(self._model):
                        batch_output = self._model(batch_input)
                    else:
                        raise RuntimeError(
                            "Model is not callable and has no forward method"
                        )

            # Process outputs
            results = self._process_batch_output(batch_output, batch_size)

            # Update metrics
            processing_time = time.time() - start_time
            self._total_inferences += batch_size
            self._total_batch_time += processing_time

            logger.debug(
                f"GPU batch processed: {batch_size} items in {processing_time:.3f}s "
                f"(throughput: {batch_size/processing_time:.1f} items/s)"
            )

            return results

        except Exception as e:
            logger.error(f"GPU batch processing failed: {e}")
            raise

    async def _prepare_batch_input(self, payloads: List[T]):
        """
        Prepare batch input for the model, handling different types.
        """
        try:
            # Check if these are summarization or embedding payloads
            if all(isinstance(p, dict) and "text" in p for p in payloads):
                # Check if these are embedding payloads (have index field)
                if all("index" in p for p in payloads):
                    return self._prepare_embedding_batch(payloads)
                else:
                    return await self._prepare_summarization_batch(payloads)

            # Check if these are clustering payloads (have embeddings field)
            elif all(isinstance(p, dict) and "embeddings" in p for p in payloads):
                return self._prepare_clustering_batch(payloads)

            else:
                # Generic payload handling
                if all(isinstance(p, torch.Tensor) for p in payloads):
                    return torch.stack(payloads)
                return payloads

        except Exception as e:
            logger.error(f"Failed to prepare batch input: {e}")
            raise

    async def _prepare_summarization_batch(self, payloads: List[dict]) -> dict:
        """
        Prepare batch input for summarization model.
        Now simplified since SummarizerUtils._summarizer handles all preprocessing.
        """
        try:
            from app.singleton import ModelManager

            # Get model components - model_loader only returns the model, so get tokenizer and device separately
            model = self._model  # Use the already loaded model
            model_manager_model, tokenizer, device = (
                ModelManager.get_summarization_model()
            )
            max_tokens = ModelManager.get_summarization_model_max_tokens()

            # Store original payloads for processing
            original_payloads = []
            for payload in payloads:
                original_payloads.append(
                    {
                        "feed": payload.get("feed"),
                        "text": payload.get("text", ""),
                        "url": payload.get("url", ""),
                        "prompt_prefix": payload.get("prompt_prefix", "summarize: "),
                    }
                )

            # Return the simplified batch structure - SummarizerUtils._summarizer will handle all preprocessing
            return {
                "original_payloads": original_payloads,
                "model": model,
                "tokenizer": tokenizer,
                "max_tokens": max_tokens,
            }

        except Exception as e:
            logger.error(f"Failed to prepare summarization batch: {e}")
            raise

    def _prepare_embedding_batch(self, payloads: List[dict]) -> dict:
        """
        Prepare batch input for embedding model.
        """
        try:
            from app.singleton import ModelManager

            # Get embedding model
            embedding_model = ModelManager.get_embedding_model()

            # Extract texts for batch processing
            texts = [payload.get("text", "") for payload in payloads]
            indices = [payload.get("index", i) for i, payload in enumerate(payloads)]

            # Generate embeddings in batch
            embeddings = embedding_model.encode(
                texts, convert_to_tensor=True, show_progress_bar=False
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                embeddings = embeddings.to(
                    torch.device(f"cuda:{self.config.device_id}")
                )

            # Convert to list format for result processing
            embedding_lists = []
            for i, embedding in enumerate(embeddings):
                embedding_list = (
                    embedding.cpu().tolist()
                    if hasattr(embedding, "cpu")
                    else embedding.tolist()
                )
                embedding_lists.append(embedding_list)

            return {
                "embeddings": embeddings,
                "indices": indices,
                "original_payloads": payloads,
            }

        except Exception as e:
            logger.error(f"Failed to prepare embedding batch: {e}")
            raise

    def _prepare_clustering_batch(self, payloads: List[dict]) -> dict:
        """
        Prepare batch input for clustering model (GPU-accelerated).
        """
        try:
            import numpy as np
            import cupy as cp

            # For clustering, we typically process one payload at a time
            # since clustering algorithms don't benefit from batching in the same way
            # as neural network inference

            # Extract the first payload (clustering is usually single-instance)
            payload = payloads[0]

            # Extract clustering parameters
            embeddings = np.array(payload.get("embeddings", []))
            use_umap = payload.get("use_umap", True)
            umap_n_components = payload.get("umap_n_components", 20)
            umap_n_neighbors = payload.get("umap_n_neighbors", 15)
            umap_min_dist = payload.get("umap_min_dist", 0.1)
            min_cluster_size = payload.get("min_cluster_size", 5)
            min_samples = payload.get("min_samples", None)
            metric = payload.get("metric", "euclidean")
            cluster_selection_method = payload.get("cluster_selection_method", "eom")
            cluster_selection_epsilon = payload.get("cluster_selection_epsilon", 0.05)
            allow_single_cluster = payload.get("allow_single_cluster", False)
            random_state = payload.get("random_state", 42)

            if len(embeddings) == 0:
                raise ValueError("Empty embeddings array provided for clustering")

            # 1) Normalize to unit sphere for cosine geometry
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if metric.lower() in ("cosine", "euclidean"):
                from sklearn.preprocessing import normalize

                embeddings = normalize(embeddings, norm="l2", axis=1, copy=False)

            # 2) Optional UMAP reduction (GPU-accelerated)
            if use_umap:
                try:
                    from cuml.manifold import UMAP as cuUMAP

                    reducer = cuUMAP(
                        n_components=umap_n_components,
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist,
                        metric="cosine",  # we normalized to unit sphere
                        random_state=random_state,
                        output_type="cupy",
                    )
                    embeddings_gpu = cp.asarray(embeddings)
                    embeddings_reduced = reducer.fit_transform(embeddings_gpu)
                    embeddings = cp.asnumpy(embeddings_reduced)
                    logger.debug(f"GPU UMAP reduction applied: {embeddings.shape}")
                except ImportError:
                    logger.warning(
                        "cuML UMAP not available, skipping GPU dimensionality reduction"
                    )
                except Exception as e:
                    logger.warning(
                        f"GPU UMAP reduction failed: {e}, using original embeddings"
                    )

            # 3) Perform GPU-accelerated clustering
            try:
                from cuml.cluster import HDBSCAN as cuHDBSCAN

                # Adjust cluster selection method for cuML HDBSCAN
                if cluster_selection_method == "eom":
                    csm = "leaf"
                else:
                    csm = cluster_selection_method

                # Use cuML HDBSCAN for GPU clustering
                embeddings_gpu = cp.asarray(embeddings)
                model = cuHDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method=csm,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    allow_single_cluster=allow_single_cluster,
                )

                labels = model.fit_predict(embeddings_gpu).astype(cp.int32)
                labels_cpu = cp.asnumpy(labels)

                logger.info(
                    f"GPU clustering completed: {len(labels_cpu)} labels, {len(set(labels_cpu) - {-1})} clusters"
                )

                return {"labels": labels_cpu, "original_payloads": payloads}

            except ImportError:
                logger.warning(
                    "cuML HDBSCAN not available, falling back to CPU clustering"
                )
                # Return payload for CPU fallback
                return {"fallback_to_cpu": True, "original_payloads": payloads}
            except Exception as e:
                logger.error(f"GPU clustering failed: {e}")
                # Return payload for CPU fallback
                return {"fallback_to_cpu": True, "original_payloads": payloads}

        except Exception as e:
            logger.error(f"Failed to prepare clustering batch: {e}")
            raise

    def _process_batch_output(self, batch_output, batch_size: int) -> List[R]:
        """
        Process batch output from the model, handling different types.

        This method handles summarization, embedding, and clustering model outputs.
        """
        try:
            logger.info(
                f"gpu_worker.py:GPUWorker:_process_batch_output batch output processing"
            )
            # Check if this is a batch with original payloads
            if isinstance(batch_output, dict) and "original_payloads" in batch_output:
                # Check if this is an embedding batch (has embeddings field)
                if "embeddings" in batch_output:
                    logger.info(
                        f"gpu_worker.py:GPUWorker:_process_batch_output embedding output processing"
                    )
                    return self._process_embedding_output(batch_output, batch_size)
                # Check if this is a clustering batch (has labels field)
                elif "labels" in batch_output:
                    logger.info(
                        f"gpu_worker.py:GPUWorker:_process_batch_output clustering output processing"
                    )
                    return self._process_clustering_output(batch_output, batch_size)
                # Check if this is a clustering fallback (fallback_to_cpu)
                elif "fallback_to_cpu" in batch_output:
                    logger.info(
                        f"gpu_worker.py:GPUWorker:_process_batch_output clustering fallback processing"
                    )
                    return self._process_clustering_fallback(batch_output, batch_size)
                else:
                    logger.info(
                        f"gpu_worker.py:GPUWorker:_process_batch_output summarization output processing"
                    )
                    return self._process_summarization_output(batch_output, batch_size)
            else:
                # Generic output processing
                if isinstance(batch_output, torch.Tensor):
                    # Split tensor into individual outputs
                    return [batch_output[i] for i in range(batch_size)]
                elif isinstance(batch_output, (list, tuple)):
                    # Already a sequence
                    return list(batch_output)
                else:
                    # Single output for batch
                    return [batch_output] * batch_size

        except Exception as e:
            logger.error(f"Failed to process batch output: {e}")
            raise

    def _process_summarization_output(
        self, batch_output: dict, batch_size: int
    ) -> List[dict]:
        """
        Process summarization model output using the same method as runtime.py.
        """
        logger.info(
            f"gpu_worker.py:GPUWorker:_process_summarization_output summarization output processing"
        )
        try:
            from app.etl.tasks import SummarizerUtils

            # Extract components from batch output
            original_payloads = batch_output["original_payloads"]
            model = batch_output["model"]
            tokenizer = batch_output["tokenizer"]
            max_tokens = batch_output["max_tokens"]

            # Get GPU device for the model - must be torch.device object for proper GPU usage
            device = torch.device(f"cuda:{self.config.device_id}")

            # Process each payload using the same summarizer method as runtime.py
            results = []
            for payload in original_payloads:
                try:
                    # Create the payload format expected by SummarizerUtils._summarizer
                    summarizer_payload = {
                        "feed": payload["feed"],
                        "text": payload["text"],
                        "url": payload["url"],
                        "prompt_prefix": "summarize: ",
                    }
                    logger.info(
                        f"GPUWorker:_process_summarization_output summarizer_payload: {summarizer_payload}"
                    )
                    # Use the same summarization method as runtime.py
                    result = SummarizerUtils._summarizer(
                        summarizer_payload, model, tokenizer, device, max_tokens
                    )
                    logger.info(
                        f"gpu_worker.py:GPUWorker:_process_summarization_output result: {result}"
                    )
                    # Extract the summary from the result
                    summary = result.get("summary", "")

                    # Create the final result in the expected format
                    final_result = {
                        "feed": payload["feed"],
                        "text": payload["text"],
                        "url": payload["url"],
                        "translated_text": result.get("translated_text", ""),
                        "compressed_text": result.get("compressed_text", ""),
                        "summary": summary,
                    }

                    results.append(final_result)

                except Exception as e:
                    logger.error(
                        f"Failed to process individual payload in GPU batch: {e}"
                    )
                    # Fallback: return original payload with error
                    fallback_result = {
                        "feed": payload["feed"],
                        "text": payload["text"],
                        "url": payload["url"],
                        "translated_text": payload.get("translated_text", ""),
                        "compressed_text": payload.get("compressed_text", ""),
                        "summary": f"Error during summarization: {str(e)}",
                    }
                    results.append(fallback_result)

            return results

        except Exception as e:
            logger.error(f"Failed to process summarization output: {e}")
            raise

    def _process_embedding_output(
        self, batch_output: dict, batch_size: int
    ) -> List[list]:
        """
        Process embedding model output.
        """
        try:
            # Extract embeddings from batch output
            embeddings = batch_output["embeddings"]
            indices = batch_output["indices"]

            # Convert embeddings to Python lists for consistency
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            else:
                return list(embeddings)

        except Exception as e:
            logger.error(f"Failed to process embedding output: {e}")
            raise

    def _process_clustering_output(
        self, batch_output: dict, batch_size: int
    ) -> List[list]:
        """
        Process clustering model output.
        """
        try:
            # Extract labels from batch output
            labels = batch_output["labels"]

            # Convert numpy array to Python list for consistency
            if hasattr(labels, "tolist"):
                return labels.tolist()
            else:
                return list(labels)

        except Exception as e:
            logger.error(f"Failed to process clustering output: {e}")
            raise

    def _process_clustering_fallback(
        self, batch_output: dict, batch_size: int
    ) -> List[dict]:
        """
        Process clustering fallback (when GPU clustering fails and falls back to CPU).
        """
        try:
            # Extract original payloads for CPU fallback
            original_payloads = batch_output["original_payloads"]

            # Return original payloads so they can be processed by CPU path
            return original_payloads

        except Exception as e:
            logger.error(f"Failed to process clustering fallback: {e}")
            raise

    async def stop(self):
        """Stop the GPU worker gracefully."""
        logger.info("gpu_worker.py:Stopping GPUWorker...")

        try:
            await self._batcher.shutdown(wait=True)

            # Clear model from GPU memory
            if self._model is not None:
                del self._model
                self._model = None

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self._is_initialized = False
            logger.info("gpu_worker.py:GPUWorker stopped successfully")

        except Exception as e:
            logger.error(f"gpu_worker.py:Error stopping GPUWorker: {e}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get GPU worker metrics."""
        batcher_metrics = self._batcher.get_metrics()

        uptime = time.time() - self._start_time
        avg_batch_time = self._total_batch_time / (
            batcher_metrics["total_batches"] or 1
        )

        return {
            **batcher_metrics,
            "gpu_device_id": self.config.device_id,
            "total_inferences": self._total_inferences,
            "uptime_seconds": uptime,
            "avg_batch_time": avg_batch_time,
            "inferences_per_second": (
                self._total_inferences / uptime if uptime > 0 else 0
            ),
            "is_initialized": self._is_initialized,
        }

    @property
    def is_initialized(self) -> bool:
        """Check if GPU worker is initialized."""
        return self._is_initialized

    @property
    def queue_depth(self) -> int:
        """Current inference queue depth."""
        return self._batcher.queue_depth
