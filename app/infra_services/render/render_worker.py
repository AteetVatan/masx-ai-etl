"""
Isolated render worker for web scraping with Playwright.

This module provides a dedicated render worker that handles all Playwright operations
in isolation to prevent event loop conflicts and ensure proper resource management.
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from .consent_resolver import ConsentResolver

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    """Configuration for the render worker."""

    # Browser settings
    browser_type: str = "chromium"  # chromium, firefox, webkit
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080

    # Performance settings
    max_concurrent_pages: int = 5
    page_timeout_ms: int = 3600000  # 1 hour for complex pages
    navigation_timeout_ms: int = 1800000  # 30 minutes for navigation

    # Resource limits
    max_memory_mb: int = 2048
    max_cpu_percent: int = 80

    # Retry settings
    max_retries: int = 3
    retry_delay_ms: int = 1000


class RenderWorker:
    """
    Isolated render worker for web scraping with Playwright.

    This class provides:
    - Single browser instance per worker
    - Concurrent page management with semaphore
    - Automatic consent dialog handling
    - Resource monitoring and cleanup
    - Error handling and retry logic
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        """
        Initialize the render worker.

        Args:
            config: Render configuration parameters
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available")

        self.config = config or RenderConfig()
        self.consent_resolver = ConsentResolver()

        # Playwright state
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

        # Concurrency control
        self._page_semaphore = asyncio.Semaphore(self.config.max_concurrent_pages)
        self._active_pages: List[Page] = []

        # Worker state
        self._is_started = False
        self._start_time = time.time()

        # Metrics
        self._total_renders = 0
        self._successful_renders = 0
        self._failed_renders = 0
        self._total_render_time = 0.0

        logger.info(
            f"render_worker.py:RenderWorker initialized: browser={self.config.browser_type}, max_pages={self.config.max_concurrent_pages}"
        )

    async def start(self):
        """Start the render worker and initialize browser."""
        if self._is_started:
            return

        try:
            logger.info("render_worker.py:Starting render worker...")

            # Initialize Playwright
            self._playwright = await async_playwright().start()

            # Launch browser
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    f"--memory-pressure-off",
                    f"--max_old_space_size={self.config.max_memory_mb}",
                ],
            )

            # Create browser context
            self._context = await self._browser.new_context(
                viewport={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                },
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )

            self._is_started = True
            logger.info("render_worker.py:Render worker started successfully")

        except Exception as e:
            logger.error(f"render_worker.py:Failed to start render worker: {e}")
            await self._cleanup()
            raise

    async def render_page(
        self, url: str, wait_for: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Render a single page with content extraction.

        Args:
            url: URL to render
            wait_for: Optional selector to wait for before extraction

        Returns:
            Dictionary containing rendered content and metadata
        """
        if not self._is_started:
            raise RuntimeError("Render worker not started")

        async with self._page_semaphore:
            return await self._render_page_internal(url, wait_for)

    async def _render_page_internal(
        self, url: str, wait_for: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal page rendering implementation."""
        page = None
        start_time = time.time()

        try:
            # Create new page
            page = await self._context.new_page()
            self._active_pages.append(page)

            # Set timeouts
            page.set_default_timeout(self.config.page_timeout_ms)
            page.set_default_navigation_timeout(self.config.navigation_timeout_ms)

            # Navigate to URL
            logger.debug(f"render_worker.py:Navigating to {url}")
            response = await page.goto(url, wait_until="networkidle")

            if not response or response.status >= 400:
                raise RuntimeError(
                    f"Failed to load page: {response.status if response else 'No response'}"
                )

            # Handle consent dialogs if needed
            if self.consent_resolver.should_handle_consent(url):
                await self._handle_consent_dialogs(page, url)

            # Wait for specific element if requested
            if wait_for:
                try:
                    await page.wait_for_selector(wait_for, timeout=5000)
                except Exception as e:
                    logger.warning(f"render_worker.py:Failed to wait for selector '{wait_for}': {e}")

            # Extract content
            content = await self._extract_page_content(page)

            # Update metrics
            render_time = time.time() - start_time
            self._total_renders += 1
            self._successful_renders += 1
            self._total_render_time += render_time

            result = {
                "url": url,
                "content": content,
                "status": "success",
                "render_time_ms": render_time * 1000,
                "timestamp": time.time(),
            }

            logger.debug(f"render_worker.py:Page rendered successfully: {url} in {render_time:.2f}s")
            return result

        except Exception as e:
            # Update metrics
            render_time = time.time() - start_time
            self._total_renders += 1
            self._failed_renders += 1
            self._total_render_time += render_time

            logger.error(f"render_worker.py:Page rendering failed for {url}: {e}")

            result = {
                "url": url,
                "content": None,
                "status": "error",
                "error": str(e),
                "render_time_ms": render_time * 1000,
                "timestamp": time.time(),
            }

            return result

        finally:
            # Clean up page
            if page:
                try:
                    self._active_pages.remove(page)
                    await page.close()
                except Exception as e:
                    logger.warning(f"render_worker.py:Error closing page: {e}")

    async def _handle_consent_dialogs(self, page: Page, url: str):
        """Handle consent dialogs and cookie banners."""
        try:
            strategy = self.consent_resolver.get_consent_strategy(url)

            if not strategy["should_handle"]:
                return

            for rule in strategy["rules"]:
                try:
                    # Wait for consent element
                    element = await page.wait_for_selector(
                        rule.selector, timeout=rule.timeout_ms
                    )

                    if element:
                        # Perform action based on rule
                        if rule.action == "click":
                            await element.click()
                        elif rule.action == "accept":
                            # Look for accept button within the element
                            accept_btn = await element.query_selector(
                                f"button:has-text('{rule.text_pattern}')"
                                if rule.text_pattern
                                else "button"
                            )
                            if accept_btn:
                                await accept_btn.click()
                        elif rule.action == "dismiss":
                            # Look for dismiss button
                            dismiss_btn = await element.query_selector(
                                f"button:has-text('{rule.text_pattern}')"
                                if rule.text_pattern
                                else "button"
                            )
                            if dismiss_btn:
                                await dismiss_btn.click()

                        # Wait a bit for the dialog to disappear
                        await asyncio.sleep(0.5)
                        break

                except Exception as e:
                    logger.debug(f"render_worker.py:Consent rule {rule.selector} failed: {e}")
                    continue

        except Exception as e:
            logger.warning(f"render_worker.py:Consent handling failed for {url}: {e}")

    async def _extract_page_content(self, page: Page) -> Dict[str, Any]:
        """Extract content from the rendered page."""
        try:
            # Get page title
            title = await page.title()

            # Get main content
            content = await page.evaluate(
                """
                () => {
                    // Try to find main content area
                    const main = document.querySelector('main, [role="main"], .main, .content, .post-content, article');
                    if (main) {
                        return main.innerText;
                    }
                    
                    // Fallback to body
                    return document.body.innerText;
                }
            """
            )

            # Get HTML for additional processing if needed
            html = await page.content()

            return {"title": title, "text": content, "html": html, "url": page.url}

        except Exception as e:
            logger.error(f"render_worker.py:Content extraction failed: {e}")
            return {
                "title": "",
                "text": "",
                "html": "",
                "url": page.url,
                "error": str(e),
            }

    async def render_multiple_pages(
        self, urls: List[str], wait_for: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Render multiple pages concurrently.

        Args:
            urls: List of URLs to render
            wait_for: Optional selector to wait for

        Returns:
            List of rendering results
        """
        if not urls:
            return []

        # Create tasks for concurrent rendering
        tasks = [self.render_page(url, wait_for) for url in urls]

        # Execute concurrently with semaphore control
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "url": urls[i],
                        "content": None,
                        "status": "error",
                        "error": str(result),
                        "timestamp": time.time(),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    async def stop(self):
        """Stop the render worker and cleanup resources."""
        if not self._is_started:
            return

        logger.info("render_worker.py:Stopping render worker...")

        try:
            # Close all active pages
            for page in self._active_pages[:]:
                try:
                    await page.close()
                except Exception as e:
                    logger.warning(f"render_worker.py:Error closing page: {e}")

            self._active_pages.clear()

            # Close browser context and browser
            if self._context:
                await self._context.close()
                self._context = None

            if self._browser:
                await self._browser.close()
                self._browser = None

            # Stop Playwright
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None

            self._is_started = False
            logger.info("render_worker.py:Render worker stopped successfully")

        except Exception as e:
            logger.error(f"render_worker.py:Error stopping render worker: {e}")
            raise

    async def _cleanup(self):
        """Emergency cleanup in case of startup failure."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.error(f"render_worker.py:Emergency cleanup failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get render worker metrics."""
        uptime = time.time() - self._start_time
        avg_render_time = self._total_render_time / (self._total_renders or 1)

        return {
            "uptime_seconds": uptime,
            "total_renders": self._total_renders,
            "successful_renders": self._successful_renders,
            "failed_renders": self._failed_renders,
            "success_rate": (
                self._successful_renders / self._total_renders
                if self._total_renders > 0
                else 0.0
            ),
            "avg_render_time_ms": avg_render_time * 1000,
            "renders_per_second": self._total_renders / uptime if uptime > 0 else 0,
            "active_pages": len(self._active_pages),
            "is_started": self._is_started,
        }

    @property
    def is_started(self) -> bool:
        """Check if render worker is started."""
        return self._is_started

    @property
    def active_page_count(self) -> int:
        """Get number of active pages."""
        return len(self._active_pages)
