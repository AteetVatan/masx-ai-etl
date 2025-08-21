# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# │  Project: MASX AI – Strategic Agentic AI System              │
# │  All rights reserved.                                        │
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

"""
Settings configuration.

Type safe configuration management using Pydantic Settings,
handling all environment variables and system configuration with proper validation.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings can be configured via environment variables or .env file.
    Type validation and defaults are handled automatically.
    """

    # Pydantic Settings to load .env file
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    require_api_key: bool = Field(
        default=True, description="Require API key for all endpoints"
    )

    # ETL_API_KEY
    etl_api_key: Optional[str] = Field(default=None, description="ETL API key")
    require_api_key: bool = Field(
        default=False, description="Require API key for all endpoints"
    )

    # Environment Configuration
    environment: str = Field(default="development", description="Environment name")

    @property
    def debug(self) -> bool:
        return self.environment == "development"

    log_level: str = Field(default="INFO", description="Logging level")

    # PROXY SETTINGS
    proxy_webpage: str = Field(
        default="https://free-proxy-list.net/", description="Proxy webpage"
    )
    proxy_testing_url: str = Field(
        default="https://httpbin.org/ip", description="Proxy testing URL"
    )
 
    # ChromaClient Settings
    chroma_dev_persist_dir: str = Field(
        default="./.chroma_storage", description="Chroma development persist directory"
    )
    chroma_prod_persist_dir: str = Field(
        default="/mnt/data/chroma", description="Chroma production persist directory"
    )

    # GDELT Settings
    gdelt_api_key: str = Field(default="", description="GDELT API key")
    gdelt_api_url: str = Field(
        default="https://api.gdeltproject.org/v2/search/gkg",
        description="GDELT API URL",
    )
    gdelt_api_keywords: str = Field(default="", description="GDELT API keywords")
    gdelt_max_records: int = Field(default=100, description="GDELT maximum records")

    # Database Configuration (Supabase)
    supabase_url: str = Field(default="", description="Supabase project URL")
    supabase_anon_key: str = Field(default="", description="Supabase anonymous key")
    supabase_service_role_key: str = Field(
        default="", description="Supabase service role key"
    )
    supabase_db_password: str = Field(
        default="", description="Supabase database password"
    )
    supabase_db_url: str = Field(default="", description="Supabase database URL")
    database_max_connections: int = Field(
        default=10, description="Maximum number of database connections"
    )
    database_min_connections: int = Field(
        default=1, description="Minimum number of database connections"
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port number")
    api_workers: int = Field(default=4, description="Number of API workers")
    
    # RunPod Worker Configuration
    runpod_workers: int = Field(default=1, description="Number of RunPod workers for parallel execution")
    
    runpod_api_key: str = Field(default="", description="RunPod API key")
    runpod_endpoint_id: str = Field(default="rrbf5aifol52jo", description="RunPod endpoint ID")
    runpod_endpoint: str = Field(default="https://api.runpod.io/v2/rrbf5aifol52jo/run", description="RunPod endpoint")
    
    
    model_pool_enabled: bool = Field(default=True, description="Enable model pool")
    model_pool_max_instances: int = Field(default=2, description="Maximum number of model instances")
    
    
    api_secret_key: str = Field(
        default="change_this_in_production", description="API secret key for security"
    )
    api_reload: bool = Field(default=False, description="API reload")

    # Performance Configuration
    max_concurrent_requests: int = Field(
        default=10, description="Maximum concurrent requests"
    )
    request_timeout: int = Field(default=7200, description="Request timeout in seconds (2 hours for ETL processes)")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: int = Field(default=2, description="Retry delay in seconds")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_memory_usage: float = Field(
        default=0.8, description="Maximum memory usage (0.0-1.0)"
    )

    # Concurrency Configuration
    # GPU settings
    gpu_batch_size: int = Field(default=32, description="GPU batch size for inference")
    gpu_max_delay_ms: int = Field(
        default=50, description="GPU max delay before batch processing (ms)"
    )
    gpu_queue_size: int = Field(default=1000, description="GPU inference queue size")
    gpu_timeout: float = Field(
        default=7200.0, description="GPU inference timeout (seconds) (2 hours for ETL processes)"
    )
    gpu_use_fp16: bool = Field(default=True, description="Use FP16 for GPU inference")
    gpu_enable_warmup: bool = Field(default=True, description="Enable GPU model warmup")

    # CPU settings
    cpu_max_threads: int = Field(default=20, description="Maximum CPU threads")
    cpu_max_processes: int = Field(default=4, description="Maximum CPU processes")

    # Render settings
    render_max_pages: int = Field(
        default=5, description="Maximum concurrent render pages"
    )
    render_queue_max: int = Field(default=100, description="Maximum render queue size")
    render_timeout_s: int = Field(default=3600, description="Render timeout (seconds) (1 hour for complex web pages)")
    render_retries: int = Field(default=3, description="Render retry attempts")

    # Security Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"], description="Allowed CORS origins"
    )
    enable_content_filtering: bool = Field(
        default=True, description="Enable content filtering"
    )
    max_content_length: int = Field(default=10000, description="Maximum content length")

    # Monitoring and Logging
    log_format: str = Field(default="json", description="Log format")
    log_file: str = Field(default="logs/masx.log", description="Log file path")
    log_rotation: str = Field(default="daily", description="Log rotation policy")
    log_retention: int = Field(default=30, description="Log retention in days")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics port")
    health_check_endpoint: str = Field(
        default="/health", description="Health check endpoint"
    )

    # Development Configuration
    enable_hot_reload: bool = Field(
        default=True, description="Enable hot reload in development"
    )
    enable_debug_toolbar: bool = Field(
        default=False, description="Enable debug toolbar"
    )
    enable_sql_logging: bool = Field(
        default=False, description="Enable SQL query logging"
    )
    test_database_url: Optional[str] = Field(
        default=None, description="Test database URL"
    )
    mock_external_apis: bool = Field(
        default=False, description="Mock external APIs in testing"
    )
    enable_api_docs: bool = Field(default=True, description="Enable GDELT integration")

    # TEST_SUMMARIZER=HDBSCANC
    test_summarizer: str = Field(default="HDBSCAN", description="Test summarizer")

    # Validators
    @field_validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_environments = ["development", "staging", "production"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v

    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level setting."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    # Computed Properties
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def has_supabase_config(self) -> bool:
        """Check if Supabase is properly configured."""
        return all(
            [self.supabase_url, self.supabase_anon_key, self.supabase_service_role_key]
        )


@lru_cache()  # Least Recently Used
def get_settings() -> Settings:
    """
    Cached to avoid reloading settings on every call.
    Settings are loaded once and reused throughout the application lifecycle.
    """
    return Settings()


# Convenience function for getting settings in other modules
def get_config() -> Settings:
    """Alias for get_settings() for backward compatibility."""
    return get_settings()
