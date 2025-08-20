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
    Synchronous version to avoid async issues on RunPod.io.
    """

    CLUSTER_TABLE_PREFIX = "news_clusters"

    def __init__(self, date: Optional[datetime] = None):
        self.db = DBOperations()
        self.logger = get_db_logger("flashpoints_cluster")
        self.cluster_table_prefix = self.CLUSTER_TABLE_PREFIX
        self.date = date

    def close(self):
        self.db.close()

    def db_cluster_init_sync(self, date: Optional[datetime] = None):
        """
        Initialize the flashpoints cluster table (synchronous version).
        """
        try:
            self.delete_news_cluster_table_sync(date)
            self.create_news_cluster_table_sync(date)
        except Exception as e:
            self.logger.error(f"Error initializing flashpoints cluster: {e}")
            raise DatabaseException(f"Error initializing flashpoints cluster: {e}")

    def create_news_cluster_table_sync(self, date: Optional[datetime] = None) -> str:
        """
        Dynamically create a daily news_clusters table if it doesn't exist (sync).
        """
        try:
            table_name = self.db.get_daily_table_name(self.cluster_table_prefix, date)

            # Create table with proper schema
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

            # Execute table creation
            self.db.execute_sync_query(create_table_query)

            # Apply RLS policies
            rls_policies = self.__get_all_rls_policies_cmd(table_name)
            for policy in rls_policies:
                self.db.execute_sync_query(policy)

            self.logger.info(f"Table created successfully: {table_name}")
            return table_name

        except Exception as e:
            self.logger.error(f"Error creating table: {e}")
            raise DatabaseException(f"Error creating table: {e}")

    def delete_news_cluster_table_sync(self, date: Optional[datetime] = None) -> str:
        """
        Dynamically delete a daily news_clusters table if it exists (sync).
        """
        try:
            table_name = self.db.get_daily_table_name(self.cluster_table_prefix, date)

            # Drop table if exists
            query = f'DROP TABLE IF EXISTS public."{table_name}";'
            self.db.execute_sync_query(query)

            self.logger.info(f"Table deleted successfully: {table_name}")
            return table_name

        except Exception as e:
            self.logger.error(f"Error deleting table: {e}")
            raise DatabaseException(f"Error deleting table: {e}")

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
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
            # Insert cluster summaries synchronously
            return self.insert_cluster_summaries_sync(flashpoint_id, clusters, date)

        except Exception as e:
            self.logger.error(
                f"[DBOperations] Error inserting clusters: {e}", exc_info=True
            )
            raise DatabaseException(f"Database cluster operation failed: {e}")

    def insert_cluster_summaries_sync(
        self,
        flashpoint_id: str,
        clusters: List[Dict[str, Any]],
        date: Optional[datetime] = None,
    ):
        """
        Insert multiple cluster summaries into the daily table (synchronous).
        """
        try:
            table_name = self.db.get_daily_table_name(self.cluster_table_prefix, date)

            # Prepare batch insert for better performance
            insert_query = f"""
            INSERT INTO "{table_name}" (
                flashpoint_id, cluster_id, summary, article_count,
                top_domains, languages, urls, images
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            # Prepare batch data
            batch_queries = []
            for cluster in clusters:
                params = (
                    str(flashpoint_id),
                    int(cluster["cluster_id"]),
                    cluster["summary"],
                    int(cluster["article_count"]),
                    json.dumps(cluster.get("top_domains", [])),
                    json.dumps(cluster.get("languages", [])),
                    json.dumps(cluster.get("urls", [])),
                    json.dumps(cluster.get("images", [])),
                )
                batch_queries.append((insert_query, params))

            # Execute batch insert
            if batch_queries:
                self.db.execute_sync_batch(batch_queries)
                self.logger.info(
                    f"Successfully inserted {len(clusters)} cluster summaries into {table_name}"
                )
            else:
                self.logger.warning("No clusters to insert")

            return True

        except Exception as e:
            self.logger.error(f"Error inserting cluster summaries: {e}")
            raise DatabaseException(f"Error inserting cluster summaries: {e}")

    def get_flashpoint_dataset_sync(
        self, date: Optional[str] = None
    ) -> List[FlashpointModel]:
        """
        Retrieve all flashpoints along with their associated feeds (synchronous).
        """
        try:
            # Fetch all flashpoints synchronously
            flashpoints = self.get_all_flashpoints_sync(date)

            # Fetch feeds for each flashpoint synchronously
            for flashpoint in flashpoints:
                feeds = self.get_feeds_per_flashpoint_sync(flashpoint.id, date)
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

    def get_all_flashpoints_sync(
        self, date: Optional[str] = None
    ) -> List[FlashpointModel]:
        """
        Retrieve all flashpoints from the daily flashpoint table (synchronous).
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

    def get_feeds_per_flashpoint_sync(
        self,
        flashpoint_id: str,
        date: Optional[str] = None,
    ) -> List[FeedModel]:
        """
        Retrieve all feeds associated with a specific flashpoint (synchronous).
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

    def __get_all_rls_policies_cmd(self, table_name: str) -> List[str]:
        """
        Generate all RLS-related SQL commands for the given table.
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
