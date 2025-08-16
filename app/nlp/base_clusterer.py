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

from app.config import get_service_logger


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
