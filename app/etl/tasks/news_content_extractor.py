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


from random import choice
import asyncio
from app.web_scrapers import BeautifulSoupExtractor, Crawl4AIExtractor
from app.singleton import ProxyManager
from app.etl_data.etl_models import FeedModel
from app.config import get_settings, get_service_logger
from app.web_scrapers import WebScraperUtils
from app.core.concurrency import CPUExecutors


class NewsContentExtractor:
    """
    Extracts raw text from article URLs using BeautifulSoup and Crawl4AI as fallback.
    """

    def __init__(self, feeds: list[FeedModel]):

        # self.news_articles = self.context.pull(DagContextEnum.NEWS_ARTICLES.value)
        self.feeds = feeds
        self.settings = get_settings()
        self.crawl4AIExtractor = Crawl4AIExtractor()
        self.logger = get_service_logger("NewsContentExtractor")

        # Initialize CPU executors for async processing
        self.cpu_executors = CPUExecutors()
        self.web_scraper_batch_size = self.settings.web_scraper_batch_size

    async def extract_feeds(self) -> list[FeedModel]:
        """
        Extract raw text for each article using proxy-enabled scraping with async concurrency.
        """
        if not self.feeds:
            self.logger.error(
                "news_content_extractor.py:NewsContentExtractor:No feeds found in context."
            )
            raise ValueError("No feeds found in context.")

        self.logger.info(
            f"news_content_extractor.py:NewsContentExtractor:---- ProxyManager initiated ----"
        )
        proxies = await ProxyManager.proxies_async()
        self.logger.info(
            f"news_content_extractor.py:NewsContentExtractor:---- {len(proxies)} proxies found ----"
        )

        if not proxies:
            self.logger.error(
                "news_content_extractor.py:NewsContentExtractor:No valid proxies found in context."
            )
            raise ValueError("No valid proxies found in context.")

        self.logger.info(
            f"news_content_extractor.py:NewsContentExtractor:-----Scraping {len(self.feeds)} feeds....Using {len(proxies)} proxies-----"
        )

        # Process feeds using async CPU executors
        scraped_feeds = []

        # Process in batches for efficiency
        batch_size = self.web_scraper_batch_size
        for i in range(0, len(self.feeds), batch_size):
            batch = self.feeds[i : i + batch_size]
            batch_results = await self._process_batch(batch, proxies)
            scraped_feeds.extend([r for r in batch_results if r])

        return scraped_feeds

    async def _process_batch(
        self, feeds: list[FeedModel], proxies: list[str]
    ) -> list[FeedModel]:
        """Process a batch of feeds using async CPU executors."""
        try:
            # Create tasks for concurrent processing
            tasks = []
            for feed in feeds:
                proxy = choice(proxies)
                task = self.cpu_executors.run_in_thread(
                    self.__scrape_multilang_feeds, feed, proxy
                )
                tasks.append(task)

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            processed_feeds = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(
                        f"news_content_extractor.py:NewsContentExtractor:Feed {i} processing failed: {result}"
                    )
                    continue

                if result and result.raw_text and len(result.raw_text) >= 1500:
                    processed_feeds.append(result)

            return processed_feeds

        except Exception as e:
            self.logger.error(
                f"news_content_extractor.py:NewsContentExtractor:Batch processing failed: {e}"
            )
            # Fallback to sequential processing
            return await self._process_batch_sequential(feeds, proxies)

    async def _process_batch_sequential(
        self, feeds: list[FeedModel], proxies: list[str]
    ) -> list[FeedModel]:
        """Fallback sequential processing."""
        results = []
        for feed in feeds:
            try:
                proxy = choice(proxies)
                result = await self.cpu_executors.run_in_thread(
                    self.__scrape_multilang_feeds, feed, proxy
                )
                if result and result.raw_text and len(result.raw_text) >= 1500:
                    results.append(result)
            except Exception as e:
                self.logger.error(
                    f"news_content_extractor.py:NewsContentExtractor:Feed processing failed: {e}"
                )
        return results

    def __scrape_multilang_feeds(self, feed: FeedModel, proxy: str) -> FeedModel:
        """
        Use BeautifulSoup first, fallback to Crawl4AI if needed.
        """
        try:

            self.logger.info(
                f"news_content_extractor.py:NewsContentExtractor:Scraping ------ {feed.url} ----- with proxy------- {proxy}"
            )

            # Step 1: Try BeautifulSoup extraction
            try:
                soup = BeautifulSoupExtractor.beautiful_soup_scrape(feed.url, proxy)
                if soup:
                    text = BeautifulSoupExtractor.extract_text_from_soup(soup)
                    if (
                        text and len(text.strip()) >= 1000
                    ):  # case sometime the JS is there to accept cookies, need a better solution.
                        feed.raw_text = text.strip()
                        feed.raw_text = WebScraperUtils.remove_links_images_ui_junk(
                            feed.raw_text
                        )
                        self.logger.info(
                            f"news_content_extractor.py:NewsContentExtractor:Successfully scraped via BeautifulSoup: {feed.url}"
                        )
                        return feed
                    else:
                        self.logger.info(
                            f"news_content_extractor.py:NewsContentExtractor:[Fallback] BeautifulSoup produced insufficient content for: {feed.url}"
                        )
                else:
                    self.logger.info(
                        f"news_content_extractor.py:NewsContentExtractor:[Fallback] BeautifulSoup failed (no soup) for: {feed.url}"
                    )
            except Exception as bs_err:
                self.logger.warning(
                    f"news_content_extractor.py:NewsContentExtractor:BeautifulSoup scraping failed for {feed.url}: {bs_err}"
                )

            # Step 2: Fallback to Crawl4AI
            self.logger.info(
                f"news_content_extractor.py:NewsContentExtractor:[Fallback] Invoking Crawl4AI for: {feed.url}"
            )
            try:
                text = asyncio.run(self.crawl4AIExtractor.crawl4ai_scrape(feed.url))
                if not text or len(text.strip()) < 1500:  # sanity check
                    raise ValueError("Crawl4AI returned empty or too short content.")
                feed.raw_text = text.strip()
                self.logger.info(
                    f"news_content_extractor.py:NewsContentExtractor:Successfully scraped via Crawl4AI: {feed.url}"
                )
                return feed
            except Exception as c4_err:
                self.logger.error(
                    f"news_content_extractor.py:NewsContentExtractor:Crawl4AI scraping failed for {feed.url}: {c4_err}"
                )
                return None

        except Exception as e:
            self.logger.error(
                f"news_content_extractor.py:NewsContentExtractor:[Error] Failed to scrape {feed.url}: {e}",
                exc_info=True,
            )
            return None


def __scrape_multilang_feeds_debug(url: str, proxy: str):
    """
    Use BeautifulSoup first, fallback to Crawl4AI if needed.
    """
    try:

        # testing save the feed model json here
        crawl4AIExtractor = Crawl4AIExtractor()

        text = ""
        # Step 1: Try BeautifulSoup extraction
        try:
            soup = BeautifulSoupExtractor.beautiful_soup_scrape(url, proxy)
            if soup:
                text = BeautifulSoupExtractor.extract_text_from_soup(soup)
                if (
                    text and len(text.strip()) >= 1000
                ):  # case sometime the JS is there to accept cookies, need a better solution.
                    text = text.strip()
                    text = WebScraperUtils.remove_links_images_ui_junk(text)

                    return text
                else:
                    print(
                        f"[Fallback] BeautifulSoup produced insufficient content for: {url}"
                    )
            else:
                print(f"[Fallback] BeautifulSoup failed (no soup) for: {url}")
        except Exception as bs_err:
            print(f"BeautifulSoup scraping failed for {url}: {bs_err}")

        # Step 2: Fallback to Crawl4AI
        print(f"[Fallback] Invoking Crawl4AI for: {url}")
        try:
            text = asyncio.run(crawl4AIExtractor.crawl4ai_scrape(url))
            if not text or len(text.strip()) < 1500:  # sanity check
                raise ValueError("Crawl4AI returned empty or too short content.")
            text = text.strip()
            print(f"Successfully scraped via Crawl4AI: {url}")
            return text
        except Exception as c4_err:
            print(f"Crawl4AI scraping failed for {url}: {c4_err}")
            return None

    except Exception as e:
        print(f"[Error] Failed to scrape {url}: {e}", exc_info=True)
        return None


# if __name__ == "__main__":

#     proxy = '198.199.86.11:8080'
#     url = 'https://news.google.com/rss/articles/CBMikgFBVV95cUxNV2xFR0ZENm1ZbEhKMUNjc0dYRkFadk5xUHBrM3ZQUlJjdjRsamNrQmFYcm1JZ1hTdkFybkRuRjVVTWxoTFA2d2Q3Vk1KZjUzcEJQdEQyd1ROUzBTRy03aVIwYnNpVFJjT0JHSmtULVlEX2tMRDhGQmdQcDRXLUQ2SjkydmgzTkRJY3VCN1lkbzdBUQ?oc=5'

#     result = __scrape_multilang_feeds(url, proxy)
#     print(result)

# import sys
# import os

# # Go 1 level up from this file (subfolder → project root)
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# sys.path.insert(0, ROOT_DIR)
