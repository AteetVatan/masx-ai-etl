"""
This schema represents a single news article.
"""

from pydantic import BaseModel, Field
from typing import Optional


class NewsArticle(BaseModel):
    """
    This schema represents a single news article.
    """

    url: str
    title: str
    raw_text: str
    summary: Optional[str] = None
    language: Optional[str] = None
    socialimage: Optional[str] = None
    domain: str
    sourcecountry: str = Field(default="unknown")
