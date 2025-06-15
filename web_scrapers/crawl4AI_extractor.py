from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from web_scrapers import WebScraperUtils


class Crawl4AIExtractor:
    """
    This class handles all Crawl4AI-related operations in the MASX AI News ETL pipeline.
    """

    @staticmethod
    async def crawl4ai_scrape(url, proxy):
        """
        Scrape the article using Crawl4AI.
        """
        prune_filter = PruningContentFilter(
            threshold=0.4,  # 0–1, lower = retain more
            threshold_type="dynamic",  # or "fixed"
            min_word_threshold=20,  # ignore small blocks
        )

        md_generator = DefaultMarkdownGenerator(
            content_filter=prune_filter, options={"ignore_links": True}
        )
        config = CrawlerRunConfig(markdown_generator=md_generator)

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config)
            if result and result.success:
                if len(result.fit_markdown) < 100:
                    # if fit_markdown is too short, use markdown
                    crawl_text = result.markdown
                else:
                    # if fit_markdown is long enough, use fit_markdown
                    crawl_text = result.fit_markdown

                text = WebScraperUtils.remove_links_images_ui_junk(crawl_text)
                return text  # Clean, relevant content

            return None
