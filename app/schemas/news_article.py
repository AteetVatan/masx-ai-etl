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
This schema represents a single news article.
"""

from typing import Optional
from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """
    This schema represents a single news article.
    """

    url: str
    url_mobile: Optional[str] = None
    title: str
    raw_text: Optional[str] = None
    summary: Optional[str] = None
    language: Optional[str] = None
    socialimage: Optional[str] = None
    domain: str
    sourcecountry: str = Field(default="unknown")
    questions: Optional[list[str]] = None
