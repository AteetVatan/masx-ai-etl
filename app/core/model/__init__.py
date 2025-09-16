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

"""
Model management package with pooling capabilities.

This package provides the new model management system that replaces the singleton
ModelManager with instance-based managers that support controlled concurrency,
memory management, and GPU/CPU optimization.
"""

from .abstract_model import AbstractModel, ModelInstance
from .model_pool_sync import ModelPool_sync
from .model_pool import ModelPool
from .model_Instance import ModelInstance
from .summarization_model_manager import SummarizationModelManager
from .summarization_finalizer_modal_manager import SummarizationFinalizerModelManager
from .embedding_model_manager import EmbeddingModelManager
from .vector_embedding_model_manager import VectorEmbeddingModelManager
from .translator_model_manager import TranslatorModelManager

__all__ = [
    "AbstractModel",
    "ModelInstance", 
    "ModelPool_sync",
    "SummarizationModelManager",
    "SummarizationFinalizerModelManager",
    "EmbeddingModelManager",
    "VectorEmbeddingModelManager",
    "TranslatorModelManager",
]
