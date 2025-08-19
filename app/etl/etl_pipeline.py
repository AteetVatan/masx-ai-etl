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

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional
from math import sqrt
import json

from app.etl.tasks import (
    NewsContentExtractor,
    Summarizer,
    VectorizeArticles,
    ClusterSummaryGenerator,
)
from app.nlp import HDBSCANClusterer, KMeansClusterer
from app.config import get_settings
from app.etl_data import Flashpoints, FlashpointsCluster
from app.etl_data.etl_models import FlashpointModel
from app.config import get_service_logger


class ETLPipeline:

    def __init__(self, date: Optional[str] = None):
        self.settings = get_settings()
        self.logger = get_service_logger("ETLPipeline")
        if date:
            self.date = date
        else:
            self.date = datetime.now().strftime("%Y-%m-%d")
        # self.date = today_date  # "2025-07-28"
        self.db_flashpoints_cluster = None

    def get_flashpoints(self, date: Optional[str] = None):
        try:
            flashpoints_service = Flashpoints()
            dataset = flashpoints_service.get_flashpoint_dataset(date)
            return dataset
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise e

    def run_all_etl_pipelines(self):
        try:           
            self.db_flashpoints_cluster = FlashpointsCluster(self.date)
            flashpoints = self.get_flashpoints(self.date)
            flashpoints = self._clean_flashpoints(flashpoints)
            if self.settings.debug:
                self.logger.info("Running ETL Pipeline in Debug Mode")
                flashpoints = [flashpoints[0]] #flashpoints[:2]
            else:
                self.logger.info("Running ETL Pipeline in Production Mode")
                flashpoints = flashpoints
            # flashpoints = [flashpoints[1]]
            # multi threading for each flashpoint
            start_time = time.time()
            self.logger.info(
                f"Starting ALL ETL Pipeline for {len(flashpoints)} flashpoints"
            )
            with ThreadPoolExecutor(max_workers=len(flashpoints)) as executor:
                futures = [
                    executor.submit(self.run_etl_pipeline, flashpoint)
                    for flashpoint in flashpoints
                ]
                results = [future.result() for future in futures]
                end_time = time.time()
                self.logger.info(
                    f"****Time taken: {end_time - start_time} seconds for ALL ETL Pipeline*****"
                )
                return results
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise e

    def run_etl_pipeline(self, flashpoint: FlashpointModel):
        try:
            if self.settings.debug:
                self.logger.info("Starting MASX News ETL (Standalone Debug Mode)")
                self.run_etl_pipeline_debug(flashpoint)
                return
            self.logger.info("Starting MASX News ETL (Standalone Production Mode)")
            start_time = time.time()
            flashpoint_id = flashpoint.id
            feeds = flashpoint.feeds

            # load summarized feeds from file
            self.logger.info("Running NewsContentExtractor...")
            extractor = NewsContentExtractor(feeds)
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
                n_clusters = round(
                    sqrt(feed_count / 2)
                )  # min(3, feed_count)  # safe default
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
                n_clusters = round(
                    sqrt(feed_count / 2)
                )  # min(5, max(2, feed_count // 2))  # dynamic fallback
                n_clusters = 3 if n_clusters < 3 else n_clusters
                clusterer = KMeansClusterer(n_clusters=n_clusters)
                cluster_summary_generator = ClusterSummaryGenerator(
                    flashpoint_id, clusterer
                )
                cluster_summaries = cluster_summary_generator.generate()

            self.logger.info(
                f" number of cluster summaries for flashpoint {flashpoint_id}: {len(cluster_summaries)}"
            )

            self.logger.info("Running db_cluster_operations...")
            # Use the synchronous version to avoid async issues
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

    def run_etl_pipeline_debug(self, flashpoint: FlashpointModel):
        print("Starting MASX News ETL (Standalone Debug Mode)")

        try:
            start_time = time.time()
            flashpoint_id = flashpoint.id

            if self.settings.test_summarizer == "HDBSCAN":
                feeds = flashpoint.feeds[:50]
            else:
                feeds = flashpoint.feeds[:10]

            # load summarized feeds from file
            if self.settings.debug and self.settings.test_summarizer == "HDBSCAN":
                summarized_feeds = self._load_summarized_feeds(flashpoint_id)
            else:
                self.logger.info("Running NewsContentExtractor...")
                extractor = NewsContentExtractor(feeds)
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
                n_clusters = round(
                    sqrt(feed_count / 2)
                )  # min(3, feed_count)  # safe default
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
                n_clusters = round(
                    sqrt(feed_count / 2)
                )  # min(5, max(2, feed_count // 2))  # dynamic fallback
                n_clusters = 3 if n_clusters < 3 else n_clusters
                clusterer = KMeansClusterer(n_clusters=n_clusters)
                cluster_summary_generator = ClusterSummaryGenerator(
                    flashpoint_id, clusterer
                )
                cluster_summaries = cluster_summary_generator.generate()

            self.logger.info("Running db_cluster_operations...")
            # Use the synchronous version to avoid async issues on RunPod.io
            self.db_flashpoints_cluster.db_cluster_operations(
                flashpoint_id, cluster_summaries, self.date
            )

            # get the time now
            end_time = time.time()
            self.logger.info(
                f"Time taken: {end_time - start_time} seconds for flashpoint {flashpoint_id}"
            )
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise e
        finally:
            self.db_flashpoints_cluster.close()
            self.logger.info("ETL Pipeline completed")

    def _clean_flashpoints(self, flashpoints: list[FlashpointModel]):
        # clean the flashpoints
        return [x for x in flashpoints if x.feeds is not None and len(x.feeds) > 0]

    def _load_summarized_feeds(self, flashpoint_id: str):
        from app.etl_data.etl_models.feed_model import FeedModel
        import json
        from pathlib import Path

        path = Path("debug_data/summarized_feed.json")

        # Best-first: UTF-8; fallback to UTF-8 with BOM; final fallback replaces bad bytes
        def load_json_textsafe(path: Path):
            for enc in ("utf-8", "utf-8-sig"):
                try:
                    with path.open("r", encoding=enc) as f:
                        return json.load(f)
                except UnicodeDecodeError:
                    continue
            # last resort: don't crash; replace undecodable bytes
            with path.open("r", encoding="utf-8", errors="replace") as f:
                return json.load(f)

        summarized_feeds_json = load_json_textsafe(path)
        summarized_feeds = [FeedModel(**feed) for feed in summarized_feeds_json]
        return summarized_feeds

    def _store_summarized_feeds(self, summarized_feeds: list[dict]):
        import json
        import re

        def clean_text(val):
            if isinstance(val, str):
                # Remove newlines and tabs
                val = val.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                # Collapse multiple spaces into one
                val = re.sub(r"\s+", " ", val).strip()
                return val
            elif isinstance(val, list):
                return [clean_text(v) for v in val]
            elif isinstance(val, dict):
                return {k: clean_text(v) for k, v in val.items()}
            return val

        cleaned_feeds = [clean_text(feed.dict()) for feed in summarized_feeds]
        json_str = json.dumps(cleaned_feeds, ensure_ascii=False, indent=4)
