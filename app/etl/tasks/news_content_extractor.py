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

"""This module handles all article-related operations in the MASX AI News ETL pipeline."""

from concurrent.futures import ThreadPoolExecutor
from random import choice
import asyncio
import concurrent.futures
from app.web_scrapers import BeautifulSoupExtractor, Crawl4AIExtractor
from app.singleton import ProxyManager
from app.etl_data.etl_models import FeedModel
from app.config import get_settings, get_service_logger
from app.web_scrapers import WebScraperUtils


class NewsContentExtractor:
    """
    Extracts raw text from article URLs using BeautifulSoup and Crawl4AI as fallback.
    """

    def __init__(self, feeds: list[FeedModel]):

        # self.news_articles = self.context.pull(DagContextEnum.NEWS_ARTICLES.value)
        self.feeds = feeds
        self.settings = get_settings()
        self.max_workers = self.settings.max_workers
        self.crawl4AIExtractor = Crawl4AIExtractor()
        self.logger = get_service_logger("NewsContentExtractor")

    def extract_feeds(self) -> list[FeedModel]:
        """
        Extract raw text for each article using proxy-enabled scraping.
        """
        if not self.feeds:
            self.logger.error("No feeds found in context.")
            raise ValueError("No feeds found in context.")

        self.logger.info(f"---- ProxyManager initiated ----")
        proxies = ProxyManager.proxies()
        self.logger.info(f"---- {len(proxies)} proxies found ----")

        if not proxies:
            self.logger.error("No valid proxies found in context.")
            raise ValueError("No valid proxies found in context.")

        self.logger.info(
            f"-----Scraping {len(self.feeds)} feeds....Using {len(proxies)} proxies-----"
        )

        scraped_feeds = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.__scrape_multilang_feeds, feed, choice(proxies))
                for feed in self.feeds
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result and result.raw_text:
                    scraped_feeds.append(result)

        return scraped_feeds

    def __scrape_multilang_feeds(self, feed: FeedModel, proxy: str) -> FeedModel:
        """
        Use BeautifulSoup first, fallback to Crawl4AI if needed.
        """
        try:
            self.logger.info(
                f"Scraping ------ {feed.url} ----- with proxy------- {proxy}"
            )

            # Step 1: Try BeautifulSoup extraction
            try:
                soup = BeautifulSoupExtractor.beautiful_soup_scrape(feed.url, proxy)
                if soup:
                    text = BeautifulSoupExtractor.extract_text_from_soup(soup)
                    if text and len(text.strip()) >= 100:
                        feed.raw_text = text.strip()
                        feed.raw_text = WebScraperUtils.remove_links_images_ui_junk(
                            feed.raw_text
                        )
                        self.logger.info(
                            f"Successfully scraped via BeautifulSoup: {feed.url}"
                        )
                        return feed
                    else:
                        self.logger.info(
                            f"[Fallback] BeautifulSoup produced insufficient content for: {feed.url}"
                        )
                else:
                    self.logger.info(
                        f"[Fallback] BeautifulSoup failed (no soup) for: {feed.url}"
                    )
            except Exception as bs_err:
                self.logger.warning(
                    f"BeautifulSoup scraping failed for {feed.url}: {bs_err}"
                )

            # Step 2: Fallback to Crawl4AI
            self.logger.info(f"[Fallback] Invoking Crawl4AI for: {feed.url}")
            try:
                text = asyncio.run(self.crawl4AIExtractor.crawl4ai_scrape(feed.url))
                if not text or len(text.strip()) < 50:  # sanity check
                    raise ValueError("Crawl4AI returned empty or too short content.")
                feed.raw_text = text.strip()
                self.logger.info(f"Successfully scraped via Crawl4AI: {feed.url}")
                return feed
            except Exception as c4_err:
                self.logger.error(f"Crawl4AI scraping failed for {feed.url}: {c4_err}")
                return None

        except Exception as e:
            self.logger.error(
                f"[Error] Failed to scrape {feed.url}: {e}", exc_info=True
            )
            return None
