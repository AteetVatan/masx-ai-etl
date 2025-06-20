import time
from etl.tasks import NewsManager, NewsContentExtractor, Summarizer


class ETLPipeline:

    @staticmethod
    def run_etl_pipeline():
        print("Starting MASX News ETL (Standalone Debug Mode)")

        try:
            start_time = time.time()

            print("\n Running NewsManager...")
            news_mgr = NewsManager()
            news_articles = news_mgr.news_articles()

            print("\n Running NewsContentExtractor...")
            extractor = NewsContentExtractor(news_articles)
            scraped_articles = extractor.extract_articles()

            print("\n Running Summarizer...")
            summarizer = Summarizer(scraped_articles)
            summarized_articles = summarizer.summarize_all_articles()
            

            # get the time now
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
        except Exception as e:
            print(f"Error: {e}")
