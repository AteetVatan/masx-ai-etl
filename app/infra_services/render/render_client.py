"""
Render client for easy access to rendering services.

This module provides a clean client API for using the render worker,
abstracting away the complexity of browser management and consent handling.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .render_worker import RenderWorker, RenderConfig

logger = logging.getLogger(__name__)


@dataclass
class RenderRequest:
    """Request for page rendering."""

    url: str
    wait_for: Optional[str] = None
    priority: int = 0
    timeout_ms: Optional[int] = None
    retry_count: int = 0


@dataclass
class RenderResponse:
    """Response from page rendering."""

    url: str
    content: Optional[Dict[str, Any]] = None
    status: str = "pending"
    error: Optional[str] = None
    render_time_ms: Optional[float] = None
    timestamp: Optional[float] = None


class RenderClient:
    """
    Client for accessing rendering services.

    This class provides a simple interface for rendering web pages,
    handling the complexity of browser management internally.
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        """
        Initialize the render client.

        Args:
            config: Render configuration parameters
        """
        self.config = config or RenderConfig()
        self._worker: Optional[RenderWorker] = None
        self._is_connected = False

        logger.info("RenderClient initialized")

    async def connect(self):
        """Connect to the render worker."""
        if self._is_connected:
            return

        try:
            self._worker = RenderWorker(config=self.config)
            await self._worker.start()
            self._is_connected = True
            logger.info("RenderClient connected to render worker")

        except Exception as e:
            logger.error(f"Failed to connect to render worker: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the render worker."""
        if not self._is_connected:
            return

        try:
            if self._worker:
                await self._worker.stop()
                self._worker = None

            self._is_connected = False
            logger.info("RenderClient disconnected from render worker")

        except Exception as e:
            logger.error(f"Error disconnecting from render worker: {e}")
            raise

    async def render_page(
        self, url: str, wait_for: Optional[str] = None
    ) -> RenderResponse:
        """
        Render a single page.

        Args:
            url: URL to render
            wait_for: Optional selector to wait for

        Returns:
            RenderResponse with content and metadata
        """
        if not self._is_connected:
            raise RuntimeError("RenderClient not connected")

        try:
            result = await self._worker.render_page(url, wait_for)

            return RenderResponse(
                url=result["url"],
                content=result.get("content"),
                status=result["status"],
                error=result.get("error"),
                render_time_ms=result.get("render_time_ms"),
                timestamp=result.get("timestamp"),
            )

        except Exception as e:
            logger.error(f"Page rendering failed for {url}: {e}")

            return RenderResponse(
                url=url,
                status="error",
                error=str(e),
                timestamp=asyncio.get_event_loop().time(),
            )

    async def render_pages(
        self, urls: List[str], wait_for: Optional[str] = None
    ) -> List[RenderResponse]:
        """
        Render multiple pages concurrently.

        Args:
            urls: List of URLs to render
            wait_for: Optional selector to wait for

        Returns:
            List of RenderResponse objects
        """
        if not urls:
            return []

        if not self._is_connected:
            raise RuntimeError("RenderClient not connected")

        try:
            results = await self._worker.render_multiple_pages(urls, wait_for)

            responses = []
            for result in results:
                response = RenderResponse(
                    url=result["url"],
                    content=result.get("content"),
                    status=result["status"],
                    error=result.get("error"),
                    render_time_ms=result.get("render_time_ms"),
                    timestamp=result.get("timestamp"),
                )
                responses.append(response)

            return responses

        except Exception as e:
            logger.error(f"Batch page rendering failed: {e}")

            # Return error responses for all URLs
            responses = []
            for url in urls:
                response = RenderResponse(
                    url=url,
                    status="error",
                    error=str(e),
                    timestamp=asyncio.get_event_loop().time(),
                )
                responses.append(response)

            return responses

    async def render_with_fallback(
        self,
        url: str,
        primary_method: str = "playwright",
        fallback_method: str = "crawl4ai",
        wait_for: Optional[str] = None,
    ) -> RenderResponse:
        """
        Render page with fallback method.

        Args:
            url: URL to render
            primary_method: Primary rendering method
            fallback_method: Fallback rendering method
            wait_for: Optional selector to wait for

        Returns:
            RenderResponse with content and metadata
        """
        # Try primary method first
        try:
            if primary_method == "playwright":
                response = await self.render_page(url, wait_for)
                if response.status == "success" and response.content:
                    return response
            # Add other methods as needed

        except Exception as e:
            logger.warning(f"Primary rendering method failed for {url}: {e}")

        # Try fallback method
        try:
            if fallback_method == "crawl4ai":
                # This would integrate with Crawl4AI
                # For now, return error response
                logger.info(f"Fallback to Crawl4AI for {url}")
                return RenderResponse(
                    url=url,
                    status="error",
                    error="Crawl4AI fallback not implemented",
                    timestamp=asyncio.get_event_loop().time(),
                )

        except Exception as e:
            logger.warning(f"Fallback rendering method failed for {url}: {e}")

        # All methods failed
        return RenderResponse(
            url=url,
            status="error",
            error="All rendering methods failed",
            timestamp=asyncio.get_event_loop().time(),
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get render client metrics."""
        if not self._worker:
            return {"status": "not_connected"}

        return {
            "client_status": "connected" if self._is_connected else "disconnected",
            "worker_metrics": self._worker.get_metrics(),
        }

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to render worker."""
        return self._is_connected

    @property
    def worker_status(self) -> str:
        """Get worker status."""
        if not self._worker:
            return "no_worker"
        return "started" if self._worker.is_started else "stopped"


# Convenience function for easy access
async def create_render_client(config: Optional[RenderConfig] = None) -> RenderClient:
    """
    Create and connect a render client.

    Args:
        config: Render configuration parameters

    Returns:
        Connected RenderClient instance
    """
    client = RenderClient(config=config)
    await client.connect()
    return client


# Context manager for automatic cleanup
class RenderClientContext:
    """Context manager for RenderClient with automatic cleanup."""

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config
        self.client: Optional[RenderClient] = None

    async def __aenter__(self) -> RenderClient:
        """Enter async context."""
        self.client = await create_render_client(self.config)
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.client:
            await self.client.disconnect()
