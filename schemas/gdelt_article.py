"""
This module contains the schemas for the GDELT articles.
"""

from pydantic import BaseModel, HttpUrl, Field, RootModel
from typing import List, Optional


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
