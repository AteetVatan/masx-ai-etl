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
"""Cluster the vectorized documents stored in ChromaDB and generate a concise summary for each cluster."""

import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from abc import ABC, abstractmethod
from typing import Optional, Callable, List

from app.config import get_service_logger, get_settings
from .base_clusterer import BaseClusterer
from app.singleton import ModelManager
from app.core.concurrency import InferenceRuntime, RuntimeConfig

import numpy as np

import hdbscan as cpu_hdbscan
from sklearn.preprocessing import normalize as sk_normalize

# Optional CPU UMAP
try:
    import umap as cpu_umap

    _HAS_CPU_UMAP = True
except Exception:
    _HAS_CPU_UMAP = False

# Optional GPU stack (RAPIDS)
_HAS_GPU_STACK = False
try:
    import cupy as cp
    from cuml.manifold import UMAP as cuUMAP
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from rmm.allocators.cupy import rmm_cupy_allocator
    import rmm

    _HAS_GPU_STACK = True
except Exception:
    pass

# Torch is only used for device discovery; the clustering uses RAPIDS/NumPy
try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


class HDBSCANClusterer(BaseClusterer):
    """
    Production HDBSCAN with CPU<->GPU auto-switch using InferenceRuntime.
    - Debug (settings.debug=True): force CPU for easy debugging.
    - Prod (settings.debug=False): use RAPIDS (GPU) if available, else CPU.
    - Text embeddings: L2 normalization (cosine geometry) + optional UMAP to 20D.

    Constructor keeps backward-compatible defaults, but the runtime improves:
      * If metric == 'euclidean' (default), we still normalize to unit sphere
        and run Euclidean (equivalent to cosine on the sphere).
      * If you explicitly pass metric='cosine', we do the same normalization and
        treat downstream distances as Euclidean on the sphere.

    Returns labels as List[int] (HDBSCAN's -1 = noise).

    HDBSCAN  (Hierarchical Density-Based Spatial Clustering of Applications with Noise) clustering strategy.
    Auto-detects number of clusters, identifies noise.
    Suitable for real-world, dense, noisy data like news.
    min_cluster_size: Minimum number of articles required to form a valid cluster.
    min_samples: Controls how conservative the clustering is. If not set, it defaults internally to the same value as min_cluster_size.
    metric: The metric to use for clustering.
    cluster_selection_method: The method to use for cluster selection.
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        use_umap: bool = True,
        umap_n_components: int = 20,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        random_state: int = 42,
        # GPU memory pool (RMM) in prod
        rmm_pool_size: str = "4GB",
        allow_single_cluster: bool = False,
        epsilon: float = 0.05,  # cluster_selection_epsilon
    ):
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = None if min_samples is None else int(min_samples)
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method

        self.settings = get_settings()
        self.device_provider = ModelManager.get_device()
        self.use_umap = bool(use_umap)
        self.umap_n_components = int(umap_n_components)
        self.umap_n_neighbors = int(umap_n_neighbors)
        self.umap_min_dist = float(umap_min_dist)
        self.random_state = int(random_state)
        self.rmm_pool_size = str(rmm_pool_size)
        self.allow_single_cluster = bool(allow_single_cluster)
        self.epsilon = float(epsilon)

        self.logger = get_service_logger("HDBSCANClusterer")

        self._debug = self.settings.debug

        # Resolve device consistently with your Torch logic
        self.device = self._resolve_device()
        self._gpu_enabled = (
            (self.device.type == "cuda") and _HAS_GPU_STACK and (not self._debug)
        )

        # Initialize inference runtime for clustering
        self.inference_runtime: Optional[InferenceRuntime] = None

        # Try to init RMM pool when GPU path is enabled
        if self._gpu_enabled:
            try:
                rmm.reinitialize(
                    pool_allocator=True, initial_pool_size=self.rmm_pool_size, devices=0
                )
                cp.cuda.set_allocator(rmm_cupy_allocator)
                self.logger.info(f"[GPU] RMM pool initialized: {self.rmm_pool_size}")
            except Exception as e:
                self.logger.warning(
                    f"[GPU] RMM init failed; will try GPU without pool or fall back. {e}"
                )

        # Set by fit; exposed via properties for optional post-processing
        self._cpu_model = None
        self._labels = None

    # ----------------- public API -----------------

    async def cluster_async(self, embeddings: np.ndarray) -> List[int]:
        """
        Cluster embeddings asynchronously using InferenceRuntime and return labels as Python ints.
        """
        try:
            # Initialize inference runtime if not already done
            if not self.inference_runtime:
                await self._initialize_inference_runtime()

            # 1) Normalize to unit sphere for cosine geometry
            X = self._normalize_for_cosine(embeddings)

            # 2) Use inference runtime for clustering
            if self._gpu_enabled:
                try:
                    # Prepare payload for GPU clustering
                    payload = {
                        "embeddings": X,
                        "use_umap": self.use_umap,
                        "umap_n_components": self.umap_n_components,
                        "umap_n_neighbors": self.umap_n_neighbors,
                        "umap_min_dist": self.umap_min_dist,
                        "min_cluster_size": self.min_cluster_size,
                        "min_samples": self.min_samples,
                        "metric": self.metric,
                        "cluster_selection_method": self.cluster_selection_method,
                        "cluster_selection_epsilon": self.epsilon,
                        "allow_single_cluster": self.allow_single_cluster,
                        "random_state": self.random_state,
                    }

                    # Use inference runtime for GPU clustering
                    result = await self.inference_runtime.infer(payload)

                    if isinstance(result, Exception):
                        self.logger.error(f"GPU clustering failed: {result}")
                        # Fallback to CPU clustering
                        return await self._cluster_cpu_async(X)

                    self._labels = result
                    self.logger.info(f"[GPU] HDBSCAN done. n={len(result)}")
                    return result.tolist()

                except Exception as ge:
                    self.logger.error(
                        f"[GPU] clustering failed. Falling back to CPU. {ge}",
                        exc_info=True,
                    )
                    # disable GPU path for this process to avoid repeated failures
                    self._gpu_enabled = False

            # CPU (debug or fallback)
            return await self._cluster_cpu_async(X)

        except Exception as e:
            self.logger.error(
                f"Error in HDBSCANClusterer.cluster_async: {e}", exc_info=True
            )
            raise

    async def _cluster_cpu_async(self, X: np.ndarray) -> List[int]:
        """CPU clustering with async wrapper."""
        try:
            # Prepare payload for CPU clustering
            payload = {
                "embeddings": X,
                "use_umap": self.use_umap,
                "umap_n_components": self.umap_n_components,
                "umap_n_neighbors": self.umap_n_neighbors,
                "umap_min_dist": self.umap_min_dist,
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "metric": self.metric,
                "cluster_selection_method": self.cluster_selection_method,
                "cluster_selection_epsilon": self.epsilon,
                "allow_single_cluster": self.allow_single_cluster,
                "random_state": self.random_state,
            }

            # Use inference runtime for CPU clustering
            result = await self.inference_runtime.infer(payload)

            if isinstance(result, Exception):
                self.logger.error(f"CPU clustering failed: {result}")
                # Fallback to synchronous CPU clustering
                return self._cluster_cpu_sync(X)

            self._labels = result
            mode = "CPU (debug)" if self._debug else "CPU (fallback)"
            self.logger.info(f"[{mode}] HDBSCAN done. n={len(result)}")
            if hasattr(result, "tolist"):
                return result.tolist()
            else:
                return list(result)

        except Exception as e:
            self.logger.error(f"CPU clustering failed: {e}")
            # Fallback to synchronous CPU clustering
            return self._cluster_cpu_sync(X)

    def cluster(self, embeddings: np.ndarray) -> List[int]:
        """
        Synchronous clustering method for backward compatibility.
        """
        import asyncio

        return asyncio.run(self.cluster_async(embeddings))

    def _cluster_cpu_sync(self, X: np.ndarray) -> List[int]:
        """Synchronous CPU clustering fallback."""
        try:
            # 1) Normalize to unit sphere for cosine geometry
            X = self._normalize_for_cosine(X)

            # 2) UMAP (CPU only)
            Z = self._reduce_cpu(X) if (self.use_umap and _HAS_CPU_UMAP) else X
            labels = self._cluster_cpu(Z)
            self._labels = labels
            mode = "CPU (debug)" if self._debug else "CPU (fallback)"
            self.logger.info(f"[{mode}] HDBSCAN done. n={len(labels)}")

            # Convert numpy array to Python list for consistency
            if hasattr(labels, "tolist"):
                return labels.tolist()
            else:
                return list(labels)
        except Exception as e:
            self.logger.error(
                f"Error in synchronous CPU clustering: {e}", exc_info=True
            )
            raise

    async def _initialize_inference_runtime(self):
        """Initialize the inference runtime for clustering."""
        try:
            # Create runtime config optimized for clustering
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
                model_loader=self._get_clustering_model_loader, config=config
            )

            await self.inference_runtime.start()
            self.logger.info("Inference runtime initialized for HDBSCAN clustering")

        except Exception as e:
            self.logger.error(f"Failed to initialize inference runtime: {e}")
            raise

    def _get_clustering_model_loader(self):
        """Model loader function for the inference runtime."""
        # Return clustering configuration for the inference runtime
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "cluster_selection_method": self.cluster_selection_method,
            "use_umap": self.use_umap,
            "umap_n_components": self.umap_n_components,
            "umap_n_neighbors": self.umap_n_neighbors,
            "umap_min_dist": self.umap_min_dist,
            "random_state": self.random_state,
            "epsilon": self.epsilon,
            "allow_single_cluster": self.allow_single_cluster,
            "gpu_enabled": self._gpu_enabled,
            "has_gpu_stack": _HAS_GPU_STACK,
            "has_cpu_umap": _HAS_CPU_UMAP,
        }

    # Optional consumers
    @property
    def probabilities_(self) -> Optional[np.ndarray]:
        # CPU-only (cuML has different APIs; not exposed here)
        if self._cpu_model is not None and hasattr(self._cpu_model, "probabilities_"):
            return self._cpu_model.probabilities_
        return None

    @property
    def outlier_scores_(self) -> Optional[np.ndarray]:
        if self._cpu_model is not None and hasattr(self._cpu_model, "outlier_scores_"):
            return self._cpu_model.outlier_scores_
        return None

    # ----------------- internals -----------------

    def _resolve_device(self):
        """
        Uses provided device_provider (Torch-style) if present.
        Else: torch.cuda.is_available() unless debug=True.
        """
        if self.device_provider is not None:
            try:
                d = self.device_provider()
                self.logger.info(f"Using device from provider: {d}")
                return d
            except Exception as e:
                self.logger.warning(
                    f"device_provider failed: {e}. Falling back to torch."
                )
        if not _HAS_TORCH:
            # Torch not available => assume CPU
            class _CPU:  # lightweight shim
                type = "cpu"

                def __str__(self):
                    return "cpu"

            self.logger.info("Torch not available; defaulting to CPU.")
            return _CPU()
        if self._debug:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {dev}")
        return dev

    def _normalize_for_cosine(self, X: np.ndarray) -> np.ndarray:
        """
        Cast to float32 and L2-normalize if metric indicates cosine/semantic geometry.
        """
        X = np.asarray(X, dtype=np.float32)
        # Treat default 'euclidean' as cosine on unit sphere for text embeddings
        # unless user explicitly sets a different non-cosine metric.
        metric = self.metric.lower()
        if metric in ("cosine", "euclidean"):
            # L2 makes euclidean == cosine on the unit sphere
            return sk_normalize(X, norm="l2", axis=1, copy=False)
        # For other metrics (e.g., manhattan), skip normalization here
        return X

    # ---------- CPU path ----------
    def _reduce_cpu(self, X: np.ndarray) -> np.ndarray:
        reducer = cpu_umap.UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric="cosine",  # we normalized to unit sphere
            random_state=self.random_state,
        )
        return reducer.fit_transform(X)

    def _cluster_cpu(self, Z: np.ndarray) -> np.ndarray:
        # Use euclidean because on the unit sphere it ≈ cosine
        metric = (
            "euclidean"
            if self.metric.lower() in ("cosine", "euclidean")
            else self.metric
        )
        csm = (
            "leaf"
            if self.cluster_selection_method == "eom"
            else self.cluster_selection_method
        )
        model = cpu_hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=metric,
            cluster_selection_method=csm,
            cluster_selection_epsilon=self.epsilon,
            allow_single_cluster=self.allow_single_cluster,
            prediction_data=True,
            approx_min_span_tree=True,
        )
        labels = model.fit_predict(Z).astype(np.int32)
        self._cpu_model = model
        return labels

    # ---------- GPU path ----------
    def _reduce_gpu(self, X: np.ndarray) -> np.ndarray:
        Xg = cp.asarray(X)
        reducer = cuUMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric="cosine",  # we normalized to unit sphere
            random_state=self.random_state,
            output_type="cupy",
        )
        Zg = reducer.fit_transform(Xg)
        return cp.asnumpy(Zg)

    def _cluster_gpu(self, Z: np.ndarray) -> np.ndarray:
        Zg = cp.asarray(Z)
        csm = (
            "leaf"
            if self.cluster_selection_method == "eom"
            else self.cluster_selection_method
        )
        # cuML HDBSCAN assumes euclidean; OK on unit sphere after normalization
        model = cuHDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=csm,
            cluster_selection_epsilon=self.epsilon,
            allow_single_cluster=self.allow_single_cluster,
        )
        labels = model.fit_predict(Zg).astype(cp.int32)
        return cp.asnumpy(labels)

    async def stop(self):
        """Stop the inference runtime."""
        if self.inference_runtime:
            await self.inference_runtime.stop()
            self.logger.info("HDBSCAN clusterer inference runtime stopped")
