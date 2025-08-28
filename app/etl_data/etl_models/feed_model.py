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

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class FeedModel(BaseModel):
    """Model for feed data."""

    id: str
    flashpoint_id: str
    url: str
    title: str
    seendate: Optional[str] = ""
    domain: Optional[str] = ""
    language: Optional[str] = ""
    sourcecountry: Optional[str] = ""
    description: Optional[str] = ""
    raw_text: Optional[str] = ""
    raw_text_en: Optional[str] = ""
    compressed_text: Optional[str] = ""
    summary: Optional[str] = ""
    image: Optional[str] = ""
    created_at: datetime = None 
    updated_at: datetime = None
