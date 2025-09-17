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


from app.config import get_service_logger, get_settings

# from app.etl.tasks import Translator
from app.etl_data.etl_models import FeedModel
from app.core.exceptions import ServiceException
from app.core.concurrency import InferenceRuntime, RuntimeConfig
from app.nlp import LanguageUtils
from typing import Optional

from app.core.concurrency import CPUExecutors
from app.enumeration import WorkloadEnums
import re
import asyncio
from app.nlp import NLPUtils


class LanguageDetectorTask:

    def __init__(self):
        self.logger = get_service_logger("LanguageDetectorTask")
        self.cpu_executors = CPUExecutors(workload=WorkloadEnums.CPU)

    async def set_language_for_all_feeds(
        self, feeds: list[FeedModel]
    ) -> list[FeedModel]:
        try:
            self.logger.info(
                f"LanguageDetectorTask:set_language_for_all_feeds {len(feeds)} feeds"
            )

            feeds_with_language = await self._set_language_for_feeds(feeds)
            return feeds_with_language
        except Exception as e:
            self.logger.error(
                f"LanguageDetectorTask:Error setting language for feeds: {e}"
            )
            raise ServiceException(f"Error setting language for feeds: {e}")
        finally:
            if self.cpu_executors:
                self.cpu_executors.shutdown(wait=True)

    async def _set_language_for_feeds(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """
        Set the language for the feeds.
        Uses thread offloading for language detection.
        """

        def get_language(txt: str) -> str:
            try:
                chunck = NLPUtils.split_text_smart(txt, 500, 1)
                txt = NLPUtils.clean_text(chunck[0])
                lang = LanguageUtils.detect_language(txt)
                return lang

            except Exception as e:
                self.logger.error(f"Language detection failed: {e}")
                return "en"

        # Process in batches for efficiency
        batch_size = self.cpu_executors.max_threads
        results = []
        for i in range(0, len(feeds), batch_size):
            batch = feeds[i : i + batch_size]
            tasks = [
                self.cpu_executors.run_in_thread(get_language, feed.processed_text)
                for feed in batch
            ]
            results.extend(await asyncio.gather(*tasks, return_exceptions=True))

        feeds_with_language: list[FeedModel] = []
        for feed, result in zip(feeds, results):
            if isinstance(result, Exception):
                self.logger.warning(f"Skipping feed {feed.id} due to error: {result}")
                continue
            if result:
                feed.language = result
                feeds_with_language.append(feed)

        return feeds_with_language
