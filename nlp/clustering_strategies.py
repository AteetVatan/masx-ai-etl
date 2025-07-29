"""Cluster the vectorized documents stored in ChromaDB and generate a concise summary for each cluster."""

import numpy as np
from sklearn.cluster import KMeans
import hdbscan
from sklearn.preprocessing import normalize
from abc import ABC, abstractmethod
from config import get_service_logger


class BaseClusterer(ABC):
    """
    Abstract base class for all clustering strategies.
    """

    @abstractmethod
    def cluster(self, embeddings: np.ndarray) -> list[int]:
        """
        Given a list of embeddings, return cluster labels.
        """
        pass


class KMeansClusterer(BaseClusterer):
    """
    KMeans clustering strategy.
    Suitable for uniform-sized, clearly separated clusters.
    """

    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.logger = get_service_logger("KMeansClusterer")

    def cluster(self, embeddings: np.ndarray) -> list[int]:
        try:
            model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            return model.fit_predict(embeddings)
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise e


class HDBSCANClusterer(BaseClusterer):
    """
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
        min_samples: int = None,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples  # Optional, fallback to default if None
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.logger = get_service_logger("HDBSCANClusterer")

    def cluster(self, embeddings: np.ndarray) -> list[int]:
        try:
            model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_method=self.cluster_selection_method,
            )
            res = model.fit_predict(embeddings)

            # embeddings = normalize(embeddings)  # L2 normalization
            # model = hdbscan.HDBSCAN(
            #     min_cluster_size=self.min_cluster_size,
            #     min_samples=self.min_samples,
            #     metric=self.metric,
            #     cluster_selection_method=self.cluster_selection_method
            # )
            # res = model.fit_predict(embeddings)

            # model = hdbscan.HDBSCAN(
            #     min_cluster_size=2,     # Allow tiny clusters
            #     min_samples=1,          # More permissive (less noise detection)
            #     metric='euclidean',     # or normalized cosine
            #     cluster_selection_method='leaf'  # leaf mode is more fine-grained
            # )
            # res = model.fit_predict(embeddings)

            return res
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise e
