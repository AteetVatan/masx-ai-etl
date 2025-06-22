import time
from etl.tasks import NewsManager, NewsContentExtractor, Summarizer, VectorizeArticles, ClusterSummaryGenerator
from nlp import HDBSCANClusterer, KMeansClusterer

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
            
            print("\n Running VectorizeArticles...")
            vectorizer = VectorizeArticles()
            collection_name = vectorizer.get_collection_name()
            vectorizer.run(summarized_articles)
            
            print("\n Running ClusterSummaryGenerator...")
            article_count = len(summarized_articles)
            if article_count < 20:
                print(f"Small dataset detected ({article_count} articles) — using KMeans clustering")
                n_clusters = min(3, article_count)  # safe default
                clusterer = KMeansClusterer(n_clusters=n_clusters)
            else:
                print(f"Dataset size ({article_count} articles) — using HDBSCAN clustering")
                clusterer = HDBSCANClusterer()
                
            cluster_summary_generator = ClusterSummaryGenerator(collection_name, clusterer)                    
            cluster_summaries = cluster_summary_generator.generate()
            
            # get the time now
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
        except Exception as e:
            print(f"Error: {e}")
