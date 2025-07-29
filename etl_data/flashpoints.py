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

from datetime import datetime
from typing import Optional, List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from etl_data.etl_models import FlashpointModel, FeedModel
from db import DBOperations
from config import get_db_logger
import asyncio


class Flashpoints:
    """
    Flashpoints class for retrieving flashpoints and their associated feeds.
    """

    def __init__(self):
        self.db = DBOperations()
        self.logger = get_db_logger("flashpoints")

    @retry(
        retry=retry_if_exception_type(Exception),  # Retry on any exception
        wait=wait_exponential(
            multiplier=1, min=2, max=10
        ),  # Exponential backoff (2s → 4s → 8s … up to 10s)
        stop=stop_after_attempt(3),  # Retry up to 3 times
        reraise=True,  # Raise the last exception if all retries fail
    )
    async def get_flashpoint_dataset(
        self, date: Optional[str] = None
    ) -> List[FlashpointModel]:
        """
        Retrieve all flashpoints along with their associated feeds.
        Retries up to 3 times in case of transient errors.
        """
        try:
            # Fetch all flashpoints
            flashpoints = await self.get_all_flashpoints(date)

            # Fetch feeds concurrently for each flashpoint
            feed_tasks = [
                self.get_feeds_per_flashpoint(fp.id, date) for fp in flashpoints
            ]
            feeds_results = await asyncio.gather(*feed_tasks)

            # Attach feeds to each flashpoint
            for flashpoint, feeds in zip(flashpoints, feeds_results):
                flashpoint.feeds = feeds

            self.logger.info(
                f"Flashpoint dataset built: {len(flashpoints)} flashpoints with feeds"
            )
            return flashpoints

        except Exception as e:
            self.logger.error(
                f"Flashpoints dataset retrieval failed: {e}", exc_info=True
            )
            raise

    async def get_all_flashpoints(
        self, date: Optional[str] = None
    ) -> List[FlashpointModel]:
        """
        Retrieve all flashpoints from the daily flashpoint table.
        Optionally filter by date (YYYY-MM-DD format).

        Args:
            date (str, optional): Date filter in YYYY-MM-DD format.

        Returns:
            List[FlashpointModel]: All flashpoints for the given date or current date.

        Raises:
            Exception: If database query or date parsing fails.
        """
        self.logger.info("Flashpoints retrieval started")

        try:
            # async Supabase client
            client = self.db.client

            # Determine table name (based on provided date or default)
            if date:
                try:
                    target_date = datetime.strptime(date, "%Y-%m-%d")
                    table_name = self.db.get_daily_table_name(
                        "flash_point", target_date
                    )
                except ValueError:
                    self.logger.error("Invalid date format. Use YYYY-MM-DD")
                    return []
            else:
                table_name = self.db.get_daily_table_name("flash_point")

            # Query all flashpoints asynchronously
            result = client.table(table_name).select("*").execute()

            # Handle empty results
            if not result.data:
                self.logger.warning("No flashpoints found")
                return []

            # Convert records into FlashpointModel instances
            flashpoints = [
                FlashpointModel(
                    id=fp.get("id", ""),
                    title=fp.get("title", ""),
                    description=fp.get("description", ""),
                    entities=self.db.parse_json_field(fp.get("entities")),
                    domains=self.db.parse_json_field(fp.get("domains")),
                    run_id=fp.get("run_id"),
                    created_at=fp.get("created_at", ""),
                    updated_at=fp.get("updated_at", ""),
                )
                for fp in result.data
            ]

            self.logger.info(f"Flashpoints retrieved: {len(flashpoints)} records")
            return flashpoints

        except Exception as e:
            self.logger.error(f"Flashpoints retrieval failed: {e}")
            raise

    async def get_feeds_per_flashpoint(
        self,
        flashpoint_id: str,
        date: Optional[str] = None,
    ) -> List[FeedModel]:
        """
        Retrieve all feeds associated with a specific flashpoint (handles Supabase 1000-row limit with pagination).

        Args:
            flashpoint_id (str): UUID of the flashpoint.
            date (str, optional): Date filter in YYYY-MM-DD format.

        Returns:
            List[FeedModel]: List of feed data for the flashpoint.

        Raises:
            Exception: If database query or date parsing fails.
        """
        self.logger.info(
            f"Feeds per flashpoint requested - flashpoint_id: {flashpoint_id}"
        )

        try:
            # async Supabase client
            client = self.db.client

            # Determine feed table (based on date or default to today)
            if date:
                try:
                    target_date = datetime.strptime(date, "%Y-%m-%d")
                    feed_table = self.db.get_daily_table_name(
                        "feed_entries", target_date
                    )
                except ValueError:
                    self.logger.error("Invalid date format. Use YYYY-MM-DD")
                    return []
            else:
                feed_table = self.db.get_daily_table_name("feed_entries")

            # Pagination setup
            all_records = []
            batch_size = 500
            offset = 0

            while True:
                result = (
                    client.table(feed_table)
                    .select("*")
                    .eq("flashpoint_id", flashpoint_id)
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                if not result.data:
                    break  # No more records to fetch

                all_records.extend(result.data)
                offset += batch_size

                # Log progress for large datasets
                self.logger.debug(
                    f"Fetched {len(result.data)} feeds (total: {len(all_records)})"
                )

                # Safety break if too many records
                if len(result.data) < batch_size:
                    break

            if not all_records:
                self.logger.warning(
                    f"No feeds found for flashpoint_id: {flashpoint_id}"
                )
                return []

            # Convert records into FeedModel instances
            feeds = [
                FeedModel(
                    id=feed.get("id", ""),
                    flashpoint_id=feed.get("flashpoint_id", ""),
                    url=feed.get("url", ""),
                    title=feed.get("title", ""),
                    seendate=feed.get("seendate"),
                    domain=feed.get("domain"),
                    language=feed.get("language"),
                    sourcecountry=feed.get("sourcecountry"),
                    description=feed.get("description"),
                    image=feed.get("image"),
                    created_at=feed.get("created_at", ""),
                    updated_at=feed.get("updated_at", ""),
                )
                for feed in all_records
            ]

            self.logger.info(
                f"Feeds retrieved for flashpoint_id={flashpoint_id}: {len(feeds)} records"
            )
            return feeds

        except Exception as e:
            self.logger.error(f"Feeds per flashpoint retrieval failed: {e}")
            raise
