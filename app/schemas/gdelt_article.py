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
This module contains the schemas for the GDELT articles.
"""

from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field, RootModel


class ArticleSchema(BaseModel):
    """
    This schema represents a single GDELT article.
    """

    url: str
    url_mobile: Optional[HttpUrl] = None
    title: str
    seendate: str  # convert this to datetime with a custom parser
    socialimage: Optional[str] = None
    domain: str
    language: str
    sourcecountry: str


class ArticleListSchema(RootModel[List[ArticleSchema]]):
    """
    This schema represents a list of GDELT articles.
    """

    pass


# class ArticleListSchema(BaseModel):
#     """
#     This schema represents a list of GDELT articles using RootModel (Pydantic v2).
#     """

#     # __root__: List[ArticleSchema]
#     pass
