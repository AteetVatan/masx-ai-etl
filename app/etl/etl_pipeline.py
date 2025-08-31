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
import asyncio
from datetime import datetime
from typing import Optional
from math import sqrt
import json
from typing import List

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
from app.core.concurrency import RunPodServerlessManager
from app.enumeration import WorkerEnums


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

    def get_flashpoints(
        self, date: Optional[str] = None, flashpoints_ids: List[str] = None
    ):
        try:
            flashpoints_service = Flashpoints()
            dataset = flashpoints_service.get_flashpoint_dataset(
                date=date, flashpoints_ids=flashpoints_ids
            )
            return dataset
        except Exception as e:
            self.logger.error(f"etl_pipeline.py:ETLPipeline:Error: {e}")
            raise e

    async def run_all_etl_pipelines(
        self,
        trigger: str = WorkerEnums.COORDINATOR.value,
        flashpoints_ids: List[str] = None,
    ):
        try:
            self.logger.info(
                f"etl_pipeline.py:ETLPipeline:**********trigger: {trigger}**********"
            )
            # initialize singletons
            # make them execute parallely and do not wait for them to complete
            if trigger == WorkerEnums.COORDINATOR.value:  # or self.settings.debug:
                self.logger.info(f"etl_pipeline.py:ETLPipeline:Coordinator trigger")
                # db table init will happen oly with coordinator
                self.db_flashpoints_cluster = FlashpointsCluster(self.date)
                self.db_flashpoints_cluster.db_cluster_init_sync(self.date)
                flashpoints = self.get_flashpoints(date=self.date)
                flashpoints = self._clean_flashpoints(flashpoints)
            elif (
                trigger == WorkerEnums.ETL_WORKER.value and flashpoints_ids is not None
            ):
                self.logger.info("etl_pipeline.py:ETLPipeline:ETL_WORKER trigger")
                # ETL_WORKER
                flashpoints = self.get_flashpoints(
                    date=self.date, flashpoints_ids=flashpoints_ids
                )
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:ETL_WORKER trigger - flashpoints length : {len(flashpoints)}"
                )
            else:
                self.logger.error(
                    f"etl_pipeline.py:ETLPipeline:Invalid trigger: {trigger}"
                )
                raise ValueError(f"Invalid trigger: {trigger}")

            if self.settings.debug:
                self.logger.info(
                    "etl_pipeline.py:ETLPipeline:Running ETL Pipeline in Debug Mode"
                )
                # flashpoints = [flashpoints[0]] #flashpoints[:2]

            start_time = time.time()
            self.logger.info(
                f"etl_pipeline.py:ETLPipeline:Starting ALL ETL Pipeline for {len(flashpoints)} flashpoints"
            )

            if trigger == WorkerEnums.COORDINATOR.value:
                # Use RunPod Serverless Manager for parallel execution
                self.logger.info(
                    "etl_pipeline.py:ETLPipeline:Running ETL Pipeline for Coordinator"
                )
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:For Coordinator -  ALL flashpoints ids"
                )
                worker_manager = RunPodServerlessManager(self.settings.runpod_workers)
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:Using {self.settings.runpod_workers} RunPod Serverless workers"
                )
                results = await worker_manager.distribute_to_workers(
                    flashpoints, date=self.date, cleanup=True
                )
            elif trigger == WorkerEnums.ETL_WORKER.value:
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:For ETL Worker - flashpoints ids: {', '.join(flashpoints_ids)}"
                )

                # return True

                tasks = [
                    self.run_etl_pipeline(flashpoint) for flashpoint in flashpoints
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            self.logger.info(
                f"etl_pipeline.py:ETLPipeline:****Time taken: {end_time - start_time} seconds for ALL ETL Pipeline*****"
            )
            return results
        except Exception as e:
            self.logger.error(f"etl_pipeline.py:ETLPipeline:Error: {e}")
            raise e

    async def run_etl_pipeline(self, flashpoint: FlashpointModel):
        try:
            # Optimize batch sizes for per-flashpoint worker isolation
            #self._optimize_batch_sizes_for_worker(flashpoint)
            
            if self.settings.debug:
                self.logger.info(
                    "etl_pipeline.py:ETLPipeline:Starting MASX News ETL (Standalone Debug Mode)"
                )
                await self.run_etl_pipeline_debug(flashpoint)
                return
            self.logger.info(
                "etl_pipeline.py:ETLPipeline:Starting MASX News ETL (Standalone Production Mode)"
            )
            start_time = time.time()
            flashpoint_id = flashpoint.id
            feeds = flashpoint.feeds

            # return True

            feeds = feeds[:5]

            # load summarized feeds from file
            self.logger.info(
                "etl_pipeline.py:ETLPipeline:Running NewsContentExtractor..."
            )
            extractor = NewsContentExtractor(feeds)
            scraped_feeds = await extractor.extract_feeds()

            self.logger.info(f"etl_pipeline.py:ETLPipeline:feeds length: {len(feeds)}")
            self.logger.info(
                f"*****************etl_pipeline.py:ETLPipeline:scraped_feeds length: {len(scraped_feeds)} out of {len(feeds)}*****************"
            )

            # scraped_feeds = scraped_feeds[:1]

            self.logger.info("etl_pipeline.py:ETLPipeline:Running Summarizer...")
            summarizer = Summarizer(scraped_feeds)
            summarized_feeds = await summarizer.summarize_all_feeds()

            self.logger.info(
                f"etl_pipeline.py:ETLPipeline:summarized_feeds length: {len(summarized_feeds)}"
            )

            for feed in summarized_feeds:
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:**************************************************************************"
                )
                self.logger.info(f"etl_pipeline.py:ETLPipeline:feed: {feed.title}")
                self.logger.info(f"etl_pipeline.py:ETLPipeline:feed: {feed.summary}")
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:**************************************************************************"
                )
            return True

            self.logger.info("etl_pipeline.py:ETLPipeline:Running VectorizeArticles...")
            vectorizer = VectorizeArticles(flashpoint_id)
            # collection_name = vectorizer.get_flashpoint_id()
            await vectorizer.run(summarized_feeds)

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
            cluster_summaries = await cluster_summary_generator.generate()

            # If HDBSCAN returns all noise
            if len(cluster_summaries) == 0:
                self.logger.info(
                    "[Clustering] HDBSCAN returned all noise. Falling back to KMeans."
                )
                n_clusters = round(
                    sqrt(feed_count / 2)
                )  # min(5, max(2, feed_count // 2))  # dynamic falldown
                n_clusters = 3 if n_clusters < 3 else n_clusters
                clusterer = KMeansClusterer(n_clusters=n_clusters)
                cluster_summary_generator = ClusterSummaryGenerator(
                    flashpoint_id, clusterer
                )
                cluster_summaries = await cluster_summary_generator.generate()

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
            self.logger.error(f"etl_pipeline.py:ETLPipeline:Error: {e}")
            raise e
        finally:
            self.db_flashpoints_cluster.close()
            self.logger.info("ETL Pipeline completed")

    async def run_etl_pipeline_debug(self, flashpoint: FlashpointModel):
        print("Starting MASX News ETL (Standalone Debug Mode)")

        try:
            start_time = time.time()
            flashpoint_id = flashpoint.id

            if self.settings.test_summarizer == "HDBSCAN":
                feeds = flashpoint.feeds[:50]
            else:
                feeds = flashpoint.feeds[:12]

            # load summarized feeds from file
            if self.settings.debug and self.settings.test_summarizer == "HDBSCAN":
                summarized_feeds = self._load_summarized_feeds(flashpoint_id)
            else:
                self.logger.info("Running NewsContentExtractor...")
                extractor = NewsContentExtractor(feeds)
                scraped_feeds = await extractor.extract_feeds()

                self.logger.info("Running Summarizer...")
                summarizer = Summarizer(scraped_feeds)
                summarized_feeds = await summarizer.summarize_all_feeds()

            self.logger.info("Running VectorizeArticles...")
            vectorizer = VectorizeArticles(flashpoint_id)
            # collection_name = vectorizer.get_flashpoint_id()
            collection_name = await vectorizer.run(summarized_feeds)

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
                collection_name, clusterer
            )
            cluster_summaries = await cluster_summary_generator.generate()

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
                cluster_summaries = await cluster_summary_generator.generate()

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
            self.logger.error(f"etl_pipeline.py:ETLPipeline:Error: {e}")
            raise e
        finally:
            self.db_flashpoints_cluster.close()
            self.logger.info("ETL Pipeline completed")

    def _clean_flashpoints(self, flashpoints: list[FlashpointModel]):
        # clean the flashpoints
        return [x for x in flashpoints if x.feeds is not None and len(x.feeds) > 0]
    
    def _optimize_batch_sizes_for_worker(self, flashpoint: FlashpointModel):
        """
        Optimize batch sizes for per-flashpoint worker isolation.
        Each worker gets dedicated RTX A4500 + 12 vCPUs + 31-62GB RAM.
        """
        if not self.settings.flashpoint_worker_enabled:
            return
        
        feed_count = len(flashpoint.feeds) if flashpoint.feeds else 0
        
        # Log worker optimization
        self.logger.info(
            f"etl_pipeline.py:ETLPipeline:Optimizing batch sizes for flashpoint worker: "
            f"Flashpoint ID: {flashpoint.id}, Feeds: {feed_count}, "
            f"RTX A4500 + 12 vCPUs + {self.settings.max_memory_usage * 100:.0f}% RAM utilization"
        )
        
        # Apply per-flashpoint worker batch multiplier
        if self.settings.is_production:
            self.logger.info(
                f"etl_pipeline.py:ETLPipeline:Per-flashpoint worker optimization enabled: "
                f"Batch multiplier: {self.settings.flashpoint_worker_batch_multiplier}x, "
                f"Max feeds per worker: {self.settings.flashpoint_worker_max_feeds}"
            )

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
