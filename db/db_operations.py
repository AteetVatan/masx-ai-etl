# # ┌───────────────────────────────────────────────────────────────┐
# # │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# # │  Project: MASX AI – Strategic Agentic AI System              │
# # │  All rights reserved.                                        │
# # └───────────────────────────────────────────────────────────────┘
# #
# # MASX AI is a proprietary software system developed and owned by Ateet Vatan Bahmani.
# # The source code, documentation, workflows, designs, and naming (including "MASX AI")
# # are protected by applicable copyright and trademark laws.
# #
# # Redistribution, modification, commercial use, or publication of any portion of this
# # project without explicit written consent is strictly prohibited.
# #
# # This project is not open-source and is intended solely for internal, research,
# # or demonstration use by the author.
# #
# # Contact: ab@masxai.com | MASXAI.com

import asyncio
import json
from datetime import datetime
from typing import Optional, List, Any

import asyncpg
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

from core.exceptions import ConfigurationException, DatabaseException
from config import get_settings, get_service_logger


class DBOperations:
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_service_logger("FlashpointDatabaseService")
        self.client: Optional[Client] = None  # Supabase client
        self.pool: Optional[asyncpg.Pool] = None  # asyncpg pool
        self._connection_params = {}

        # Initialize DB parameters
        self._initialize_connection()

        # Auto-connect on instantiation (sync wrapper)
        asyncio.run(self._sync_initialize_connections())
        
    async def get_new_connection(self):
        return await asyncpg.connect(self._connection_params["database_url"])

    def _initialize_connection(self):
        """Initialize database connection parameters."""
        try:
            if not self.settings.supabase_url or not self.settings.supabase_anon_key:
                raise ConfigurationException("Supabase URL and key are required")

            self._connection_params = {
                "supabase_url": self.settings.supabase_url,
                "supabase_key": self.settings.supabase_anon_key,
                "database_url": self.settings.supabase_db_url,
                "max_connections": self.settings.database_max_connections or 10,
                "min_connections": self.settings.database_min_connections or 1,
            }

            self.logger.info("Database connection parameters initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {e}")
            raise DatabaseException(f"Database initialization failed: {str(e)}")

    async def connect(self):
        """Establish async DB connections."""
        try:
            # Initialize Supabase client
            options = ClientOptions(schema="public", headers={"X-Client-Info": "masx-ai-system"})
            self.client = create_client(
                self._connection_params["supabase_url"],
                self._connection_params["supabase_key"],
                options=options,
            )

            # Initialize asyncpg pool if DB URL exists
            database_url = self._connection_params.get("database_url")
            if database_url:
                self.pool = await asyncpg.create_pool(
                    database_url,
                    min_size=self._connection_params["min_connections"],
                    max_size=self._connection_params["max_connections"],
                    command_timeout=60,
                    server_settings={"application_name": "masx_ai_system"},
                )

            self.logger.info("Database connections established successfully")

        except Exception as e:
            self.logger.error(f"Failed to establish database connections: {e}")
            raise DatabaseException(f"Database connection failed: {str(e)}")

    async def _sync_initialize_connections(self):
        """Internal helper for initializing connections in sync context."""
        await self.connect()

    async def disconnect(self):
        """Close async DB connections."""
        try:
            if self.pool:
                await self.pool.close()
                self.pool = None

            if self.client:
                self.client = None

            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")

    def close(self):
        """Sync wrapper for disconnect."""
        asyncio.run(self.disconnect())

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    def get_client(self) -> Client:
        """Sync accessor for Supabase client."""
        if not self.client:
            asyncio.run(self._sync_initialize_connections())
        return self.client

    def get_pool(self) -> asyncpg.Pool:
        """Sync accessor for asyncpg pool."""
        if not self.pool:
            asyncio.run(self._sync_initialize_connections())
        return self.pool

    @staticmethod
    def get_daily_table_name(base: str, date: Optional[datetime] = None) -> str:
        """Generate daily table name (base_YYYYMMDD)."""
        if date is None:
            date_obj = datetime.utcnow()
        elif isinstance(date, str):
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid date format: {date}. Expected 'YYYY-MM-DD'.")
        elif isinstance(date, datetime):
            date_obj = date
        else:
            raise TypeError(f"Unsupported date type: {type(date)}. Must be datetime, str, or None.")
        return f"{base}_{date_obj.strftime('%Y%m%d')}"

    @staticmethod
    def parse_json_field(field_value: Any) -> List[str]:
        """Safely parse JSON field that might be string or list."""
        if isinstance(field_value, str):
            try:
                return json.loads(field_value)
            except json.JSONDecodeError:
                return []
        elif isinstance(field_value, list):
            return field_value
        return []
