"""Cluster the vectorized documents stored in ChromaDB and generate a concise summary for each cluster."""

import numpy as np
from sklearn.cluster import KMeans
import hdbscan
from abc import ABC, abstractmethod


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

    def cluster(self, embeddings: np.ndarray) -> list[int]:
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        return model.fit_predict(embeddings)


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

    def __init__(self, min_cluster_size: int = 5, min_samples: int = None, metric: str = "euclidean", cluster_selection_method: str = "eom"):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples  # Optional, fallback to default if None
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method

    def cluster(self, embeddings: np.ndarray) -> list[int]:
        model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method
        )
        return model.fit_predict(embeddings)
