"""
This module handles all web scraping operations in the MASX AI News ETL pipeline.
"""

from .beautiful_soap_extractor import BeautifulSoupExtractor
from .crawl4AI_extractor import Crawl4AIExtractor
from .web_scraper_utils import WebScraperUtils

__all__ = ["BeautifulSoupExtractor", "Crawl4AIExtractor", "WebScraperUtils"]
