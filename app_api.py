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
FastAPI application for Global Signal Grid (MASX) Agentic AI System.

Main application setup with:
- Middleware configuration
- Error handling
- Route registration
- CORS and security
- Request/response logging
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog

from config.settings import get_settings
from config.logging_config import get_api_logger
from core.exceptions import (
    MASXException,
    ConfigurationException,
    DatabaseException,
    TranslationException,
    EmbeddingException,
    AgentException,
)
from api.routes import services


async def verify_api_key(request: Request):
    """
    Verify API key from request headers.

    Args:
        request: FastAPI request object

    Returns:
        bool: True if API key is valid

    Raises:
        HTTPException: If API key is missing or invalid
    """
    settings = get_settings()

    # Skip verification if not required
    if not settings.require_api_key:
        return True

    # Get API key from headers
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Please provide X-API-Key or Authorization header",
        )

    # Remove 'Bearer ' prefix if present
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]

    # Verify against configured API key
    if api_key != settings.etl_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger = get_api_logger("AppLifespan")

    # Startup
    logger.info("Starting MASX AI System API")
    try:
        # Initialize services here if needed
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"API startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down MASX AI System API")
    try:
        # Cleanup services here if needed
        logger.info("API shutdown completed successfully")
    except Exception as e:
        logger.error(f"API shutdown failed: {e}")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    settings = get_settings()
    logger = get_api_logger("AppFactory")

    # Create FastAPI app with global dependencies
    app = FastAPI(
        title="MASX AI-GlobalSignalGrid API",
        description="Global Signal Grid (GSG) Agentic AI System API",
        version="1.0.0",
        docs_url="/docs" if settings.enable_api_docs else None,
        redoc_url="/redoc" if settings.enable_api_docs else None,
        openapi_url="/openapi.json" if settings.enable_api_docs else None,
        lifespan=lifespan,
        dependencies=[Depends(verify_api_key)] if settings.require_api_key else None,
    )

    # Add middleware
    _add_middleware(app, settings)

    # Add exception handlers
    _add_exception_handlers(app)

    # Add request/response logging
    _add_logging_middleware(app)

    # Register routes
    _register_routes(app)

    logger.info("FastAPI application created successfully")
    return app


def _add_middleware(app: FastAPI, settings):
    """Add middleware to the application."""
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response


def _add_exception_handlers(app: FastAPI):
    """Add exception handlers to the application."""

    @app.exception_handler(MASXException)
    async def masx_exception_handler(request: Request, exc: MASXException):
        """Handle MASX-specific exceptions."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "MASX-GlobalSignalGrid System Error",
                "message": str(exc),
                "type": exc.__class__.__name__,
                "path": request.url.path,
            },
        )

    @app.exception_handler(ConfigurationException)
    async def configuration_exception_handler(
        request: Request, exc: ConfigurationException
    ):
        """Handle configuration exceptions."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Configuration Error",
                "message": str(exc),
                "type": "ConfigurationException",
                "path": request.url.path,
            },
        )

    @app.exception_handler(DatabaseException)
    async def database_exception_handler(request: Request, exc: DatabaseException):
        """Handle database exceptions."""
        return JSONResponse(
            status_code=503,
            content={
                "error": "Database Error",
                "message": str(exc),
                "type": "DatabaseException",
                "path": request.url.path,
            },
        )

    @app.exception_handler(TranslationException)
    async def translation_exception_handler(
        request: Request, exc: TranslationException
    ):
        """Handle translation exceptions."""
        return JSONResponse(
            status_code=503,
            content={
                "error": "Translation Error",
                "message": str(exc),
                "type": "TranslationException",
                "path": request.url.path,
            },
        )

    @app.exception_handler(EmbeddingException)
    async def embedding_exception_handler(request: Request, exc: EmbeddingException):
        """Handle embedding exceptions."""
        return JSONResponse(
            status_code=503,
            content={
                "error": "Embedding Error",
                "message": str(exc),
                "type": "EmbeddingException",
                "path": request.url.path,
            },
        )

    @app.exception_handler(AgentException)
    async def agent_exception_handler(request: Request, exc: AgentException):
        """Handle agent exceptions."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Agent Error",
                "message": str(exc),
                "type": "AgentException",
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger = get_api_logger("ExceptionHandler")
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "type": "Exception",
                "path": request.url.path,
            },
        )


def _add_logging_middleware(app: FastAPI):
    """Add request/response logging middleware."""
    logger = get_api_logger("RequestLogging")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests and responses."""
        start_time = time.time()

        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        # Process request
        try:
            response = await call_next(request)

            # Log response
            process_time = time.time() - start_time
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=process_time,
            )

            # Add process time header
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                process_time=process_time,
            )
            raise


def _register_routes(app: FastAPI):
    """Register API routes."""
    # Health check routes
    # app.include_router(health.router, prefix="/health", tags=["Health"])

    # Workflow routes
    # app.include_router(workflows.router, prefix="/workflows", tags=["Workflows"])

    # Data routes
    # app.include_router(data.router, prefix="/data", tags=["Data"])

    # Service routes
    app.include_router(services.router, prefix="/services", tags=["Services"])

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "MASX AI System API",
            "version": "1.0.0",
            "description": "Global Signal Grid (GSG) Agentic AI System",
            "status": "operational",
            "endpoints": {
                # "health": "/health",
                # "workflows": "/workflows",
                # "data": "/data",
                "services": "/services",
                "docs": "/docs",
            },
        }
