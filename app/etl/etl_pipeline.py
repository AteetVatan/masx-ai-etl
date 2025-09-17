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
    SummarizerTask,
    SummarizerFinalizerTask,
    VectorizeTask,
    ClusterSummaryGenerator,
)
from app.nlp import HDBSCANClusterer, KMeansClusterer
from app.config import get_settings
from app.etl_data import Flashpoints, FlashpointsCluster
from app.etl_data.etl_models import FlashpointModel, ClusterModel
from app.config import get_service_logger
from app.core.concurrency import RunPodServerlessManager
from app.enumeration import WorkerEnums
from app.etl.tasks import CompressorTask
from app.etl.tasks import TranslatorTask
from app.etl.tasks import LanguageDetectorTask
from app.etl.etl_pipeline_debug import ETLPipelineDebug


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
            flashpoint_id = flashpoint.id
            feeds = flashpoint.feeds

            if self.settings.debug:
                feeds = feeds[:5]

            start_time = time.time()
            if True:

                # load summarized feeds from file
                self.logger.info(
                    f"\n\n=============================NewsContentExtractor=====================================\n\n"
                )

                self.logger.info(
                    "etl_pipeline.py:ETLPipeline:Running NewsContentExtractor..."
                )

                extractor = NewsContentExtractor()
                processed_feeds = await extractor.extract_feeds(feeds)
                news_content_extractor_time = time.time() - start_time
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:NewsContentExtractor time: {news_content_extractor_time} seconds"
                )

                self.logger.info(
                    f"*****************etl_pipeline.py:ETLPipeline:scraped_feeds length: {len(processed_feeds)} out of {len(feeds)}*****************"
                )

                self.logger.info(
                    f"\n\n=====================================================================================\n\n"
                )

                self.logger.info(
                    f"\n\n=============================LanguageDetectorTask====================================\n\n"
                )

                language_detector_time = time.time()
                # Set the language for the compressed feeds
                language_detector = LanguageDetectorTask()
                processed_feeds = await language_detector.set_language_for_all_feeds(
                    processed_feeds
                )
                language_detector_time = time.time() - language_detector_time
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:LanguageDetector time: {language_detector_time} seconds"
                )

                self.logger.info(
                    f"\n\n=====================================================================================\n\n"
                )

                self.logger.info(
                    f"\n\n=============================CompressorTask==========================================\n\n"
                )
                compressor_time = time.time()
                compressor = CompressorTask()
                processed_feeds = await compressor.compress_all_feeds(processed_feeds)
                compressor_time = time.time() - compressor_time
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:Compressor time: {compressor_time} seconds"
                )

                self.logger.info(
                    f"\n\n=====================================================================================\n\n"
                )

                self.logger.info(
                    f"\n\n=============================TranslatorTask==========================================\n\n"
                )
                translator_time = time.time()
                # Translate the compressed feeds
                translator = TranslatorTask()
                processed_feeds = await translator.translate_all_feeds(processed_feeds)
                translator_time = time.time() - translator_time
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:Translator time: {translator_time} seconds"
                )

                self.logger.info(
                    f"\n\n=====================================================================================\n\n"
                )
                # summarize the processed feeds
                self.logger.info(
                    f"\n\n=============================SummarizerTask==========================================\n\n"
                )
                summarizer_time = time.time()
                self.logger.info("etl_pipeline.py:ETLPipeline:Running Summarizer...")
                summarizer = SummarizerTask()
                processed_feeds = await summarizer.summarize_all_feeds(processed_feeds)
                summarizer_time = time.time() - summarizer_time
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:Summarizer time: {summarizer_time} seconds"
                )

                self.logger.info(
                    f"\n\n=====================================================================================\n\n"
                )

                if self.settings.debug:
                    #### Here store the summarize processed_feeds as json file
                    store_time = time.time()
                    self.logger.info(
                        "etl_pipeline.py:ETLPipeline:Storing summarized feeds to JSON file..."
                    )
                    stored_file_path = ETLPipelineDebug.store_summarized_feeds(
                        processed_feeds, flashpoint_id, self.date
                    )
                    store_time = time.time() - store_time
                    self.logger.info(
                        f"etl_pipeline.py:ETLPipeline:Store time: {store_time} seconds"
                    )

            #### then read the json file to create the processed_feeds again
            if self.settings.debug:
                load_time = time.time()
                self.logger.info(
                    "etl_pipeline.py:ETLPipeline:Loading summarized feeds from JSON file..."
                )
                processed_feeds = ETLPipelineDebug.load_summarized_feeds(
                    flashpoint_id, self.date
                )
                load_time = time.time() - load_time
                self.logger.info(
                    f"etl_pipeline.py:ETLPipeline:Load time: {load_time} seconds"
                )

            if not processed_feeds:
                self.logger.error(
                    "etl_pipeline.py:ETLPipeline:Failed to load summarized feeds from JSON file"
                )
                raise ValueError("Failed to load summarized feeds from JSON file")

            # summarize the processed feeds
            # summarizer_finalizer_time = time.time()
            # self.logger.info("etl_pipeline.py:ETLPipeline:Running SummarizerFinalizer...")
            # summarizer_finalizer = SummarizerFinalizerTask()
            # summarized_feeds = await summarizer_finalizer.summarize_all_feeds_finalizer(processed_feeds)
            # summarizer_finalizer_time = time.time() - summarizer_finalizer_time
            # self.logger.info(f"etl_pipeline.py:ETLPipeline:SummarizerFinalizer time: {summarizer_finalizer_time} seconds")

            self.logger.info(
                f"etl_pipeline.py:ETLPipeline:summarized_feeds length: {len(processed_feeds)}"
            )

            # for feed in processed_feeds[:5]:
            #     self.logger.info(
            #         f"etl_pipeline.py:ETLPipeline:**************************************************************************"
            #     )
            #     self.logger.info(f"etl_pipeline.py:ETLPipeline:feed: {feed.title}")
            #     self.logger.info(f"etl_pipeline.py:ETLPipeline:feed: {feed.summary}")
            #     self.logger.info(
            #         f"etl_pipeline.py:ETLPipeline:**************************************************************************"
            #     )

            self.logger.info(
                f"\n\n=============================VectorizeTask==========================================\n\n"
            )
            vectorize_time = time.time()
            self.logger.info("etl_pipeline.py:ETLPipeline:Running VectorizeArticles...")
            vectorizer = VectorizeTask(flashpoint_id)
            # collection_name = vectorizer.get_flashpoint_id()
            vectorized_collection_name = await vectorizer.run(processed_feeds)
            vectorize_time = time.time() - vectorize_time
            self.logger.info(
                f"etl_pipeline.py:ETLPipeline:VectorizeTask time: {vectorize_time} seconds"
            )
            self.logger.info(
                f"\n\n=====================================================================================\n\n"
            )

            self.logger.info(
                f"\n\n=============================ClusterSummaryGenerator=================================\n\n"
            )

            self.logger.info("Running ClusterSummaryGenerator...")
            feed_count = len(processed_feeds)

            if feed_count < 50:
                self.logger.info(
                    f"Small dataset detected ({feed_count} articles) — using KMeans clustering"
                )
                n_clusters = round(
                    sqrt(feed_count / 2)
                )  # min(3, feed_count)  # safe default
                clusterer = KMeansClusterer()
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
                cluster_summaries: list[ClusterModel] = (
                    await cluster_summary_generator.generate()
                )

            self.logger.info(
                f" number of cluster summaries for flashpoint {flashpoint_id}: {len(cluster_summaries)}"
            )
            self.logger.info(
                f"\n\n============================================================================================\n\n"
            )

            self.logger.info(
                f"\n\n=============================db_cluster_operations==========================================\n\n"
            )

            self.logger.info("Running db_cluster_operations...")
            # Use the synchronous version to avoid async issues
            db_cluster_operations_time = time.time()
            self.db_flashpoints_cluster = FlashpointsCluster(self.date)
            self.db_flashpoints_cluster.db_cluster_operations(
                flashpoint_id, cluster_summaries, self.date
            )
            db_cluster_operations_time = time.time() - db_cluster_operations_time
            self.logger.info(
                f"etl_pipeline.py:ETLPipeline:db_cluster_operations time: {db_cluster_operations_time} seconds"
            )
            self.logger.info(
                f"\n\n============================================================================================\n\n"
            )

            # get the time now
            end_time = time.time()
            self.logger.info(f"Time taken: {end_time - start_time} seconds")
            self.logger.info(
                f"\n\n============================================================================================\n\n"
            )
        except Exception as e:
            self.logger.error(f"etl_pipeline.py:ETLPipeline:Error: {e}")
            raise e
        finally:
            if self.db_flashpoints_cluster:
                self.db_flashpoints_cluster.close()
            self.logger.info("ETL Pipeline completed")

    def _clean_flashpoints(self, flashpoints: list[FlashpointModel]):
        # clean the flashpoints
        return [x for x in flashpoints if x.feeds is not None and len(x.feeds) > 0]
