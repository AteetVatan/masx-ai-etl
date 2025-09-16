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

from dataclasses import dataclass
from typing import Any, Optional, TypeVar, Generic
import torch


T = TypeVar('T')


@dataclass
class ModelInstance(Generic[T]):
    """Container for a model instance with metadata."""
    
    model: T
    tokenizer: Optional[Any] = None
    device: Optional[torch.device] = None
    model_type: str = ""
    in_use: bool = False
    created_at: float = 0.0
    vram_usage_bytes: int = 0
    max_tokens: int = 0
    
    # Hash & equality by identity (so it can live in a set)
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other
    
    def destroy(self) -> None:
        """Clean up model resources."""
        if hasattr(self.model, 'cpu'):
            self.model.cpu()
        if hasattr(self.model, 'delete'):
            self.model.delete()
        del self.model
        if self.tokenizer:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
