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

from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from .feed_model import FeedModel


class FlashpointModel(BaseModel):
    """Model for flashpoint data."""

    id: str
    title: str
    description: str
    entities: List[str]
    domains: List[str]
    feeds: Optional[List[FeedModel]] = []
    run_id: Optional[str]
    created_at: datetime
    updated_at: datetime
