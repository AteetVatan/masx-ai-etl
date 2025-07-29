import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from etl.tasks import (
    NewsManager,
    NewsContentExtractor,
    Summarizer,
    VectorizeArticles,
    ClusterSummaryGenerator,
)
from nlp import HDBSCANClusterer, KMeansClusterer
from config import get_settings
from etl_data import Flashpoints, FlashpointsCluster
from etl_data.etl_models import FlashpointModel, FeedModel
from config import get_service_logger
from typing import Optional
from datetime import datetime


class ETLPipeline:

    def __init__(self, date: Optional[str] = None):
        self.settings = get_settings()
        self.logger = get_service_logger("ETLPipeline")
        if date:
            self.date = date
        else:
            self.date = datetime.now().strftime("%Y-%m-%d")
        #self.date = today_date  # "2025-07-28"
        self.db_flashpoints_cluster = FlashpointsCluster(self.date)

    def get_flashpoints(self, date: Optional[str] = None):
        try:
            flashpoints_service = Flashpoints()
            dataset = asyncio.run(flashpoints_service.get_flashpoint_dataset(date))
            return dataset
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise e

    def run_all_etl_pipelines(self):
        try:
            flashpoints = self.get_flashpoints(self.date)
            # flashpoints = flashpoints[:1]
            # flashpoints = [flashpoints[1]]
            # multi threading for each flashpoint
            with ThreadPoolExecutor(max_workers=len(flashpoints)) as executor:
                futures = [
                    executor.submit(self.run_etl_pipeline, flashpoint)
                    for flashpoint in flashpoints
                ]
                results = [future.result() for future in futures]
                return results
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise e

    def run_etl_pipeline(self, flashpoint: FlashpointModel):
        print("Starting MASX News ETL (Standalone Debug Mode)")

        try:
            start_time = time.time()
            flashpoint_id = flashpoint.id
            self.logger.info("Running NewsContentExtractor...")
            extractor = NewsContentExtractor(flashpoint.feeds)
            scraped_feeds = extractor.extract_feeds()

            self.logger.info("Running Summarizer...")
            summarizer = Summarizer(scraped_feeds)
            summarized_feeds = summarizer.summarize_all_feeds()

            self.logger.info("Running VectorizeArticles...")
            vectorizer = VectorizeArticles(flashpoint_id)
            # collection_name = vectorizer.get_flashpoint_id()
            vectorizer.run(summarized_feeds)

            self.logger.info("Running ClusterSummaryGenerator...")
            feed_count = len(summarized_feeds)
            if feed_count < 50:
                self.logger.info(
                    f"Small dataset detected ({feed_count} articles) — using KMeans clustering"
                )
                n_clusters = min(3, feed_count)  # safe default
                clusterer = KMeansClusterer(n_clusters=n_clusters)
            else:
                self.logger.info(
                    f"Dataset size ({feed_count} articles) — using HDBSCAN clustering"
                )
                clusterer = HDBSCANClusterer()

            cluster_summary_generator = ClusterSummaryGenerator(
                flashpoint_id, clusterer
            )
            cluster_summaries = cluster_summary_generator.generate()

            # If HDBSCAN returns all noise
            if len(cluster_summaries) == 0:
                self.logger.info(
                    "[Clustering] HDBSCAN returned all noise. Falling back to KMeans."
                )
                n_clusters = min(5, max(2, feed_count // 2))  # dynamic fallback
                clusterer = KMeansClusterer(n_clusters=n_clusters)
                cluster_summary_generator = ClusterSummaryGenerator(
                    flashpoint_id, clusterer
                )
                cluster_summaries = cluster_summary_generator.generate()

            self.db_flashpoints_cluster.db_cluster_operations(
                flashpoint_id, cluster_summaries, self.date
            )

            # get the time now
            end_time = time.time()
            self.logger.info(f"Time taken: {end_time - start_time} seconds")
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise e
        finally:
            self.db_flashpoints_cluster.close()
            self.logger.info("ETL Pipeline completed")
