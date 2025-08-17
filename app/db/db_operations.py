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
from typing import Optional, List, Any

# import asyncpg  # Commented out to prevent async issues
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

# Add synchronous PostgreSQL support
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None

from app.core.exceptions import ConfigurationException, DatabaseException
from app.config import get_settings, get_service_logger

# Add connection pooling for synchronous operations
try:
    from psycopg2.pool import SimpleConnectionPool
    POOL_AVAILABLE = True
except ImportError:
    POOL_AVAILABLE = False
    SimpleConnectionPool = None


class DBOperations:
    """
    DBOperations class for managing database connections.
    """

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_service_logger("FlashpointDatabaseService")
        self.client: Optional[Client] = None  # Supabase client
        self.pool = None  # asyncpg pool (disabled)
        self.sync_pool: Optional[SimpleConnectionPool] = None  # psycopg2 pool
        self._connection_params = {}

        # Initialize DB parameters (synchronous only)
        self._initialize_connection()

        # DO NOT initialize any async connections during __init__
        # This prevents the TypeError: __init__() should return None, not '_asyncio.Task'
        self.logger.info("DBOperations initialized (async connections deferred)")

    # async def get_new_connection(self):
    #     return await asyncpg.connect(self._connection_params["database_url"])

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

            # DO NOT initialize connection pools during __init__ to prevent async issues
            # Pools will be initialized lazily when first needed
            self.logger.info("Database connection parameters initialized (pools deferred)")

        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {e}")
            raise DatabaseException(f"Database initialization failed: {str(e)}")

    def _initialize_sync_pool(self):
        """Initialize synchronous connection pool."""
        try:
            if not POOL_AVAILABLE:
                return
                
            database_url = self._connection_params.get("database_url")
            if not database_url:
                return
            
            # Create connection pool
            self.sync_pool = SimpleConnectionPool(
                minconn=self._connection_params["min_connections"],
                maxconn=self._connection_params["max_connections"],
                dsn=database_url,
                cursor_factory=RealDictCursor
            )
            
            self.logger.info("Synchronous connection pool initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize synchronous connection pool: {e}")
            # Don't fail initialization if pool creation fails
            self.sync_pool = None

    # async def connect(self):
    #     """Establish async DB connections."""
    #     try:
    #         # Initialize Supabase client
    #         options = ClientOptions(
    #             schema="public", headers={"X-Client-Info": "masx_ai-system"}
    #         )
    #         self.client = create_client(
    #             self._connection_params["supabase_url"],
    #             self._connection_params["supabase_key"],
    #             options=options,
    #         )

    #         # Initialize asyncpg pool if DB URL exists
    #         database_url = self._connection_params.get("database_url")
    #         if database_url:
    #             self.pool = await asyncpg.create_pool(
    #                 database_url,
    #                 min_size=self._connection_params["min_connections"],
    #                 max_size=self._connection_params["max_connections"],
    #                 command_timeout=60,
    #                 server_settings={"application_name": "masx_ai_system"},
    #             )

    #         self.logger.info("Database connections established successfully")

    #     except Exception as e:
    #         self.logger.error(f"Failed to establish database connections: {e}")
    #         raise DatabaseException(f"Database connection failed: {str(e)}")

    # async def _sync_initialize_connections(self):
    #     """Internal helper for initializing connections in sync context."""
    #     await self.connect()

    # async def disconnect(self):
    #     """Close async DB connections."""
    #     try:
    #         if self.pool:
    #             await self.pool.close()
    #             self.pool = None

    #         if self.client:
    #             self.client = None

    #         self.logger.info("Database connections closed")
    #     except Exception as e:
    #         self.logger.error(f"Error closing database connections: {e}")

    def close(self):
        """Close database connections."""
        try:
            # Close synchronous connection pool
            if self.sync_pool:
                self.sync_pool.closeall()
                self.sync_pool = None
                self.logger.info("Synchronous connection pool closed")
        except Exception as e:
            self.logger.warning(f"Error closing synchronous connection pool: {e}")
        
        # Note: Async connections are not automatically closed to avoid async issues
        # They will be cleaned up when the process ends
        self.logger.info("Database operations closed (async connections preserved)")

    # async def __aenter__(self):
    #     """Async context manager entry."""
    #     await self.connect()
    #     return self

    # async def __aexit__(self, exc_type, exc_val, exc_tb):
    #     """Async context manager exit."""
    #     await self.disconnect()

    def get_client(self) -> Client:
        """Sync accessor for Supabase client."""
        if not self.client:
            # Note: Client initialization is deferred to avoid async issues
            self.logger.warning("Supabase client not initialized - async operations disabled")
        return self.client

    def get_pool(self):
        """Sync accessor for asyncpg pool."""
        if not self.pool:
            # Note: Pool initialization is deferred to avoid async issues
            self.logger.warning("Async pool not initialized - async operations disabled")
        return self.pool

    def get_sync_connection(self):
        """
        Get a synchronous database connection using psycopg2.
        Returns a connection object that can be used for synchronous operations.
        """
        if not PSYCOPG2_AVAILABLE:
            raise DatabaseException("psycopg2 is not available. Install it with: pip install psycopg2-binary")
        
        try:
            database_url = self._connection_params.get("database_url")
            if not database_url:
                raise DatabaseException("Database URL not configured")
            
            # Initialize sync pool lazily if not already done
            if not self.sync_pool and POOL_AVAILABLE:
                self._initialize_sync_pool()
            
            # Try to get connection from pool first
            if self.sync_pool:
                try:
                    conn = self.sync_pool.getconn()
                    if conn:
                        # Test connection
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                        cursor.fetchone()
                        cursor.close()
                        
                        self.logger.debug("Synchronous database connection obtained from pool")
                        return conn
                except Exception as e:
                    self.logger.warning(f"Pool connection failed, falling back to direct connection: {e}")
                    # Return connection to pool if it's broken
                    if conn:
                        self.sync_pool.putconn(conn, close=True)
            
            # Fallback to direct connection
            conn = psycopg2.connect(
                database_url,
                cursor_factory=RealDictCursor,
                application_name="masx_ai_system_sync"
            )
            
            # Set autocommit to False for transaction control
            conn.autocommit = False
            
            # Test connection
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            
            self.logger.debug("Synchronous database connection established and validated")
            return conn
            
        except Exception as e:
            self.logger.error(f"Failed to establish synchronous database connection: {e}")
            raise DatabaseException(f"Synchronous connection failed: {str(e)}")

    def return_sync_connection(self, conn):
        """
        Return a connection to the pool if using connection pooling.
        
        Args:
            conn: Database connection to return
        """
        if self.sync_pool and conn:
            try:
                self.sync_pool.putconn(conn)
                self.logger.debug("Connection returned to pool")
            except Exception as e:
                self.logger.warning(f"Failed to return connection to pool: {e}")
                # Close connection if pool return fails
                try:
                    conn.close()
                except:
                    pass
        elif conn:
            # Close direct connection
            try:
                conn.close()
            except:
                pass

    def test_sync_connection(self) -> bool:
        """
        Test if synchronous database connection is working.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            conn = self.get_sync_connection()
            conn.close()
            self.logger.info("Synchronous database connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Synchronous database connection test failed: {e}")
            return False

    def execute_sync_query(self, query: str, params: tuple = None, fetch: bool = False):
        """
        Execute a synchronous database query with proper error handling.
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            fetch: Whether to fetch and return results
            
        Returns:
            Query results if fetch=True, otherwise None
        """
        conn = None
        try:
            conn = self.get_sync_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                results = cursor.fetchall()
                return results
            else:
                conn.commit()
                return None
                
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            self.logger.error(f"Query execution failed: {e}")
            raise DatabaseException(f"Query execution failed: {str(e)}")
        finally:
            if conn:
                cursor.close()
                self.return_sync_connection(conn)

    def execute_sync_batch(self, queries: List[tuple]):
        """
        Execute multiple queries in a single transaction.
        
        Args:
            queries: List of (query, params) tuples
            
        Returns:
            List of results for each query
        """
        conn = None
        try:
            conn = self.get_sync_connection()
            cursor = conn.cursor()
            results = []
            
            for query, params in queries:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Try to fetch results if available
                try:
                    result = cursor.fetchall()
                    results.append(result)
                except psycopg2.ProgrammingError:
                    # No results to fetch (INSERT, UPDATE, DELETE)
                    results.append(None)
            
            conn.commit()
            return results
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            self.logger.error(f"Batch execution failed: {e}")
            raise DatabaseException(f"Batch execution failed: {str(e)}")
        finally:
            if conn:
                cursor.close()
                self.return_sync_connection(conn)

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
            raise TypeError(
                f"Unsupported date type: {type(date)}. Must be datetime, str, or None."
            )
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
