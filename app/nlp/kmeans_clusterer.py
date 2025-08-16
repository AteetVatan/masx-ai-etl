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

from sklearn.cluster import KMeans
from app.config import get_service_logger
import numpy as np
from .base_clusterer import BaseClusterer


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
