"""
This module handles all web scraping operations in the MASX AI News ETL pipeline.
"""

from .beautiful_soap_extractor import BeautifulSoupExtractor
from .web_scraper_utils import WebScraperUtils
from .crawl4AI_extractor import Crawl4AIExtractor


__all__ = ["BeautifulSoupExtractor", "WebScraperUtils", "Crawl4AIExtractor"]
