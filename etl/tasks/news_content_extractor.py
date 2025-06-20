"""
This module handles all article-related operations in the MASX AI News ETL pipeline.
"""

from concurrent.futures import ThreadPoolExecutor
from random import choice
import asyncio
import concurrent.futures
from web_scrapers import BeautifulSoupExtractor, Crawl4AIExtractor
from singleton import ProxyManager, EnvManager
from enums import EnvKeyEnum
from schemas import ArticleListSchema, NewsArticle


class NewsContentExtractor:
    """
    Extracts raw text from article URLs using BeautifulSoup and Crawl4AI as fallback.
    """

    def __init__(self, news_articles: list[NewsArticle]):

        # self.news_articles = self.context.pull(DagContextEnum.NEWS_ARTICLES.value)
        self.news_articles = news_articles
        self.env_config = EnvManager.get_env_vars()
        self.max_workers = self.env_config[EnvKeyEnum.MAX_WORKERS.value]
        self.Crawl4AIExtractor = Crawl4AIExtractor()

    def extract_articles(self) -> list[NewsArticle]:
        """
        Extract raw text for each article using proxy-enabled scraping.
        """
        if not self.news_articles:
            raise ValueError("No news articles found in context.")
        

        proxies = ProxyManager.proxies()
        if not proxies:
            raise ValueError("No valid proxies found in context.")

        print(f"-----Scraping {len(self.news_articles)} articles....Using {len(proxies)} proxies-----")

        scraped_articles = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self.__scrape_multilang_article, article, choice(proxies)
                )
                for article in self.news_articles
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result and result.raw_text:
                    scraped_articles.append(result)

        return scraped_articles

    def __scrape_multilang_article(
        self, article: NewsArticle, proxy: str
    ) -> NewsArticle:
        """
        Use BeautifulSoup first, fallback to Crawl4AI if needed.
        """
        try:
            print(f"Scraping ------ {article.url} ----- with proxy------- {proxy}")
            soup = BeautifulSoupExtractor.beautiful_soup_scrape(article.url, proxy)
            if not soup:
                return None

            text = BeautifulSoupExtractor.extract_text_from_soup(soup)
            if not text or len(text.strip()) < 100:
                print(f"[Fallback] Using Crawl4AI for: {article.url}")
                text = asyncio.run(self.Crawl4AIExtractor.crawl4ai_scrape(article.url))

            article.raw_text = text
            return article

        except Exception as e:
            print(f"[Error] Failed to scrape {article.url}: {e}")
            return None
