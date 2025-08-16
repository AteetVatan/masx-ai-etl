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

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

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


class FlashpointsCluster:
    """
    Flashpoints class for retrieving flashpoints and their associated feeds.
    """

    CLUSTER_TABLE_PREFIX = "news_clusters"

    def __init__(self, date: Optional[datetime] = None):
        self.db = DBOperations()
        self.logger = get_db_logger("flashpoints_cluster")
        self.cluster_table_prefix = self.CLUSTER_TABLE_PREFIX

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # For environments like FastAPI
            return asyncio.create_task(self._db_cluster_init(date))
        else:
            return loop.run_until_complete(self._db_cluster_init(date))

    def close(self):
        self.db.close()

    async def _db_cluster_init(self, date: Optional[datetime] = None):
        """
        Initialize the flashpoints cluster table (async version).
        """
        try:
            await self.delete_news_cluster_table(date)
            await self.create_news_cluster_table(date)
        except Exception as e:
            self.logger.error(f"Error initializing flashpoints cluster: {e}")
            raise DatabaseException(f"Error initializing flashpoints cluster: {e}")

    async def create_news_cluster_table(self, date: Optional[datetime] = None) -> str:
        """
        Dynamically create a daily news_clusters table if it doesn't exist.
        """
        # conn = await self.db.get_new_connection()  # asyncpg connection for DDL
        try:
            await self.db.connect()
            pool = self.db.get_pool()
            if not pool:
                raise DatabaseException("PostgreSQL pool not initialized")

            table_name = self.db.get_daily_table_name(self.cluster_table_prefix, date)
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                id BIGSERIAL PRIMARY KEY,
                flashpoint_id UUID NOT NULL,
                cluster_id INT NOT NULL,
                summary TEXT NOT NULL,
                article_count INT NOT NULL,
                top_domains JSONB DEFAULT '[]'::jsonb,
                languages JSONB DEFAULT '[]'::jsonb,
                urls JSONB DEFAULT '[]'::jsonb,
                images JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
            rls_policies = self.__get_all_rls_policies_cmd(table_name)

            async with pool.acquire() as conn:
                await conn.execute(create_table_query)
                for policy in rls_policies:
                    await conn.execute(policy)
                self.logger.info(f"Table ensured: {table_name}")

        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise DatabaseException(f"Error: {e}")
        finally:
            await self.db.disconnect()

    async def delete_news_cluster_table(self, date: Optional[datetime] = None) -> str:
        """
        Dynamically delete a daily news_clusters table if it exists using asyncpg.
        """
        try:
            await self.db.connect()
            pool = self.db.get_pool()
            if not pool:
                raise DatabaseException("PostgreSQL pool not initialized")
            table_name = self.db.get_daily_table_name(self.cluster_table_prefix, date)
            query = f'DROP TABLE IF EXISTS public."{table_name}";'  # Use quotes to handle special characters
            async with pool.acquire() as conn:
                await conn.execute(query)
            self.logger.info(f"Deleted table: {table_name}")
            return table_name
        except Exception as e:
            self.logger.error(f"Error deleting table: {e}")
            raise DatabaseException(f"Error deleting table: {e}")
        finally:
            await self.db.disconnect()

    @retry(
        retry=retry_if_exception_type(Exception),  # Retry on any exception
        wait=wait_exponential(
            multiplier=1, min=2, max=10
        ),  # Exponential backoff (2s → 4s → 8s … up to 10s)
        stop=stop_after_attempt(3),  # Retry up to 3 times
        reraise=True,  # Raise the last exception if all retries fail
    )
    def db_cluster_operations(
        self,
        flashpoint_id: str,
        clusters: List[Dict[str, Any]],
        date: Optional[datetime] = None,
    ):
        """
        Synchronous method to create the daily cluster table and insert cluster summaries.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                # For environments like FastAPI
                return asyncio.create_task(
                    self.insert_cluster_summaries(flashpoint_id, clusters, date)
                )
            else:
                return loop.run_until_complete(
                    self.insert_cluster_summaries(flashpoint_id, clusters, date)
                )
        except Exception as e:
            self.logger.error(
                f"[DBOperations] Error inserting clusters: {e}", exc_info=True
            )
            raise DatabaseException(f"Database cluster operation failed: {e}")

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def insert_cluster_summaries(
        self,
        flashpoint_id: str,
        clusters: List[Dict[str, Any]],
        date: Optional[datetime] = None,
    ):
        """
        Insert multiple cluster summaries into the daily table.

        Args:
            clusters: List of cluster summary dicts.
            date: Optional date for daily table.
        """
        try:
            table_name = self.db.get_daily_table_name(self.cluster_table_prefix, date)
            await self.db.connect()
            client = self.db.get_client()  # Supabase client for DML
            if not client:
                raise DatabaseException("Supabase client not connected")

            payload = [
                {
                    "flashpoint_id": str(flashpoint_id),  # ensure string
                    "cluster_id": int(c["cluster_id"]),  # convert np.int32 → int
                    "summary": c["summary"],
                    "article_count": int(c["article_count"]),  # convert np.int32 → int
                    "top_domains": json.dumps(c.get("top_domains", [])),
                    "languages": json.dumps(c.get("languages", [])),
                    "urls": json.dumps(c.get("urls", [])),
                    "images": json.dumps(c.get("images", [])),
                }
                for c in clusters
            ]

            result = client.table(table_name).insert(payload).execute()
            print(result)
            self.logger.info(
                f"Inserted {len(clusters)} cluster summaries into {table_name}"
            )
        except Exception as e:
            self.logger.error(f"Error: {e}")
            raise DatabaseException(f"Error: {e}")
        finally:
            await self.db.disconnect()

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
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
            await self.db.connect()
            client = self.db.get_client()  # Supabase client for DML

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

            # Query all flashpoints asynchronously via Supabase client
            result = client.table(table_name).select("*").execute()

            if not result.data:
                self.logger.warning("No flashpoints found")
                return []

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
        finally:
            await self.db.disconnect()

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
            await self.db.connect()
            client = self.db.get_client()  # Supabase client for DML

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
                    break

                all_records.extend(result.data)
                offset += batch_size

                self.logger.debug(
                    f"Fetched {len(result.data)} feeds (total: {len(all_records)})"
                )

                if len(result.data) < batch_size:
                    break

            if not all_records:
                self.logger.warning(
                    f"No feeds found for flashpoint_id: {flashpoint_id}"
                )
                return []

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
        finally:
            await self.db.disconnect()

    def __get_all_rls_policies_cmd(self, table_name: str) -> List[str]:
        """
        Generate all RLS-related SQL commands for the given table:
        - Enables and forces RLS
        - Creates SELECT, INSERT, UPDATE, DELETE policies for 'authenticated' role
        """

        enable_rls_query = f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY;"
        force_rls_query = f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY;"

        create_select_policy_query = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE policyname = 'allow_select' AND tablename = '{table_name}'
            ) THEN
                EXECUTE format($sql$
                    CREATE POLICY allow_select ON {table_name}
                    FOR SELECT TO anon, authenticated USING (true);
                $sql$);
            END IF;
        END $$;
        """

        create_insert_policy_query = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE policyname = 'allow_insert' AND tablename = '{table_name}'
            ) THEN
                EXECUTE format($sql$
                    CREATE POLICY allow_insert ON {table_name}
                    FOR INSERT TO anon, authenticated WITH CHECK (true);
                $sql$);
            END IF;
        END $$;
        """

        create_update_policy_query = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE policyname = 'allow_update' AND tablename = '{table_name}'
            ) THEN
                EXECUTE format($sql$
                    CREATE POLICY allow_update ON {table_name}
                    FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);
                $sql$);
            END IF;
        END $$;
        """

        create_delete_policy_query = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE policyname = 'allow_delete' AND tablename = '{table_name}'
            ) THEN
                EXECUTE format($sql$
                    CREATE POLICY allow_delete ON {table_name}
                    FOR DELETE TO anon, authenticated USING (true);
                $sql$);
            END IF;
        END $$;
        """

        return [
            enable_rls_query,
            force_rls_query,
            create_select_policy_query,
            create_insert_policy_query,
            create_update_policy_query,
            create_delete_policy_query,
        ]
