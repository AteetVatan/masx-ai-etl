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
from app.etl_data.etl_models import FlashpointModel, FeedModel
from app.db import DBOperations
from app.config import get_db_logger
from app.core.exceptions import DatabaseException


class Flashpoints:
    """
    Flashpoints class for retrieving flashpoints and their associated feeds.
    Synchronous version to avoid async issues on RunPod.io.
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
    def get_flashpoint_dataset(
        self, date: Optional[str] = None
    ) -> List[FlashpointModel]:
        """
        Retrieve all flashpoints along with their associated feeds (synchronous).
        Retries up to 3 times in case of transient errors.
        """
        try:
            # Fetch all flashpoints
            flashpoints = self.get_all_flashpoints(date)

            # Fetch feeds for each flashpoint synchronously
            for flashpoint in flashpoints:
                feeds = self.get_feeds_per_flashpoint(flashpoint.id, date)
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

    def get_all_flashpoints(self, date: Optional[str] = None) -> List[FlashpointModel]:
        """
        Retrieve all flashpoints from the daily flashpoint table (synchronous).
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
            # Determine table name
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

            # Query all flashpoints
            query = f'SELECT * FROM "{table_name}"'
            results = self.db.execute_sync_query(query, fetch=True)

            if not results:
                self.logger.warning("No flashpoints found")
                return []

            # Convert to FlashpointModel instances
            flashpoints = []
            for fp in results:
                try:
                    flashpoint = FlashpointModel(
                        id=fp.get("id", ""),
                        title=fp.get("title", ""),
                        description=fp.get("description", ""),
                        entities=self.db.parse_json_field(fp.get("entities")),
                        domains=self.db.parse_json_field(fp.get("domains")),
                        run_id=fp.get("run_id"),
                        created_at=fp.get("created_at", ""),
                        updated_at=fp.get("updated_at", ""),
                    )
                    flashpoints.append(flashpoint)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to parse flashpoint {fp.get('id')}: {e}"
                    )
                    continue

            self.logger.info(f"Flashpoints retrieved: {len(flashpoints)} records")
            return flashpoints

        except Exception as e:
            self.logger.error(f"Flashpoints retrieval failed: {e}")
            raise

    def get_feeds_per_flashpoint(
        self,
        flashpoint_id: str,
        date: Optional[str] = None,
    ) -> List[FeedModel]:
        """
        Retrieve all feeds associated with a specific flashpoint (synchronous).
        Handles Supabase 1000-row limit with pagination.

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
            # Determine feed table name
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

            # Query feeds with pagination to handle large datasets
            all_records = []
            batch_size = 500
            offset = 0

            while True:
                query = f"""
                SELECT * FROM "{feed_table}" 
                WHERE flashpoint_id = %s 
                ORDER BY created_at 
                LIMIT %s OFFSET %s
                """
                params = (flashpoint_id, batch_size, offset)

                result = self.db.execute_sync_query(query, params, fetch=True)

                if not result:
                    break

                all_records.extend(result)
                offset += batch_size

                self.logger.debug(
                    f"Fetched {len(result)} feeds (total: {len(all_records)})"
                )

                if len(result) < batch_size:
                    break

            if not all_records:
                self.logger.warning(
                    f"No feeds found for flashpoint_id: {flashpoint_id}"
                )
                return []

            # Convert to FeedModel instances
            feeds = []
            for feed in all_records:
                try:
                    feed_model = FeedModel(
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
                    feeds.append(feed_model)
                except Exception as e:
                    self.logger.warning(f"Failed to parse feed {feed.get('id')}: {e}")
                    continue

            self.logger.info(
                f"Feeds retrieved for flashpoint_id={flashpoint_id}: {len(feeds)} records"
            )
            return feeds

        except Exception as e:
            self.logger.error(f"Feeds per flashpoint retrieval failed: {e}")
            raise
