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
Logging configuration.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import structlog
from structlog.stdlib import LoggerFactory

from .settings import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """
    logging for the GlobalSignalGrid system.
    log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file: Path to log file
    log_format: Log format ('json' or 'text')

    This function sets up:
    Structured logging with JSON output
    **Log rotation and retention
    **Console and file handlers
    **Custom formatters for different environments
    """
    settings = get_settings()

    # Use provided params or fall back to settings
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file
    log_format = log_format or settings.log_format

    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            (
                structlog.processors.JSONRenderer()
                if log_format == "json"
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        file_handler = create_rotating_file_handler(
            log_file, settings.log_rotation, settings.log_retention
        )
        root_logger.addHandler(file_handler)

    # Set specific loggers
    configure_third_party_loggers(log_level)

    # Log startup message
    logger = structlog.get_logger(__name__)
    logger.info(
        "Logging configured",
        log_level=log_level,
        log_file=log_file,
        log_format=log_format,
        environment=settings.environment,
    )


def create_rotating_file_handler(
    log_file: str, rotation: str, retention: int
) -> logging.handlers.RotatingFileHandler:
    """
    Create a rotating file handler for log files.
    log_file: Path to the log file
    rotation: Rotation policy ('daily', 'weekly', 'monthly')
    retention: Number of backup files to keep

    Returns:  Configured rotating file handler
    """
    # Calculate max bytes based on rotation policy
    if rotation == "daily":
        max_bytes = 10 * 1024 * 1024  # 10MB
        backup_count = retention
    elif rotation == "weekly":
        max_bytes = 50 * 1024 * 1024  # 50MB
        backup_count = retention
    elif rotation == "monthly":
        max_bytes = 100 * 1024 * 1024  # 100MB
        backup_count = retention
    else:
        max_bytes = 10 * 1024 * 1024  # Default 10MB
        backup_count = 30  # Default 30 backups

    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )

    handler.setLevel(logging.DEBUG)  # File handler captures all levels

    return handler


def configure_third_party_loggers(log_level: str) -> None:
    """
    Configure logging levels for third-party libraries.
    Args: log_level: Base logging level for the application
    """
    # Set levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("langgraph").setLevel(logging.INFO)
    logging.getLogger("supabase").setLevel(logging.INFO)
    logging.getLogger("psycopg2").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    # Set application loggers to DEBUG in development
    if get_settings().is_development:
        logging.getLogger("app").setLevel(logging.DEBUG)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    name: Logger name (usually __name__)
    Returns: Configured structured logger
    """
    return structlog.get_logger(name)


def log_agent_action(
    logger: structlog.stdlib.BoundLogger,
    agent_name: str,
    action: str,
    parameters: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    status: str = "success",
    error: Optional[str] = None,
    workflow_id: Optional[str] = None,
) -> None:
    """
    Log a standardized agent action for audit purposes.
    logger: Structured logger instance
    agent_name: Name of the agent performing the action
    action: Action being performed
    parameters: Input parameters for the action
    result: Output result from the action
    status: Status of the action ('success', 'failure', 'warning')
    error: Error message if action failed
    workflow_id: Unique identifier for the current run

    This function provides consistent logging format for all agent actions,
    enabling comprehensive audit trails and monitoring.
    """
    log_data = {
        "agent": agent_name,
        "action": action,
        "status": status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if workflow_id:
        log_data["workflow_id"] = workflow_id

    if parameters:
        log_data["parameters"] = parameters

    if result:
        log_data["result"] = result

    if error:
        log_data["error"] = error
        log_data["status"] = "failure"

    if status == "success":
        logger.info("Agent action completed", **log_data)
    elif status == "failure":
        logger.error("Agent action failed", **log_data)
    elif status == "warning":
        logger.warning("Agent action warning", **log_data)
    else:
        logger.info("Agent action", **log_data)


def log_workflow_step(
    logger: structlog.stdlib.BoundLogger,
    step_name: str,
    step_type: str,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    duration: Optional[float] = None,
    workflow_id: Optional[str] = None,
) -> None:
    """
    Log a workflow step execution.
    logger: Structured logger instance
    step_name: Name of the workflow step
    step_type: Type of step ('agent', 'service', 'validation', etc.)
    input_data: Input data for the step
    output_data: Output data from the step
    duration: Execution duration in seconds
    workflow_id: Unique identifier for the current run
    """
    log_data = {
        "step_name": step_name,
        "step_type": step_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if workflow_id:
        log_data["workflow_id"] = workflow_id

    if input_data:
        log_data["input"] = input_data

    if output_data:
        log_data["output"] = output_data

    if duration is not None:
        log_data["duration_seconds"] = duration

    logger.info("Workflow step executed", **log_data)


def log_system_event(
    logger: structlog.stdlib.BoundLogger,
    event_type: str,
    event_data: Optional[Dict[str, Any]] = None,
    severity: str = "info",
) -> None:
    """
    Log system-level events.
    logger: Structured logger instance
    event_type: Type of system event
    event_data: Additional event data
    severity: Event severity ('info', 'warning', 'error', 'critical')
    """
    log_data = {
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if event_data:
        log_data.update(event_data)

    if severity == "info":
        logger.info("System event", **log_data)
    elif severity == "warning":
        logger.warning("System warning", **log_data)
    elif severity == "error":
        logger.error("System error", **log_data)
    elif severity == "critical":
        logger.critical("System critical", **log_data)
    else:
        logger.info("System event", **log_data)


def get_service_logger(service_name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger for a service module.
    service_name: Name of the service (e.g., 'DatabaseService')
    Returns: Configured structured logger
    """
    return structlog.get_logger(f"service.{service_name}")

def get_db_logger(db_name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger for a database module.
    db_name: Name of the database (e.g., 'Database')
    Returns: Configured structured logger
    """
    return structlog.get_logger(f"db.{db_name}")


def get_api_logger(api_name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger for a api module.
    api_name: Name of the api (e.g., 'api')
    Returns: Configured structured logger
    """
    return structlog.get_logger(f"api.{api_name}")
