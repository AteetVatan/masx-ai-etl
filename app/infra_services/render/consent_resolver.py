"""
Consent resolver for web scraping automation.

This module handles cookie banners, consent dialogs, and other UI elements
that commonly block web scraping operations.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConsentRule:
    """Rule for handling consent dialogs."""

    selector: str
    action: str  # "click", "accept", "reject", "dismiss"
    text_pattern: Optional[str] = None
    timeout_ms: int = 5000
    required: bool = False


class ConsentResolver:
    """
    Automated consent dialog resolver for web scraping.

    This class provides strategies for handling common consent dialogs,
    cookie banners, and other UI elements that block content access.
    """

    def __init__(self):
        """Initialize consent resolver with common rules."""
        self.common_rules = self._get_common_rules()
        self.logger = logger

    def _get_common_rules(self) -> List[ConsentRule]:
        """Get common consent handling rules."""
        return [
            # Cookie banners
            ConsentRule(
                selector="[id*='cookie'], [class*='cookie'], [data-testid*='cookie']",
                action="accept",
                text_pattern="accept|agree|allow|ok|got it",
                timeout_ms=3000,
            ),
            ConsentRule(
                selector="[id*='consent'], [class*='consent']",
                action="accept",
                text_pattern="accept|agree|allow|ok",
                timeout_ms=3000,
            ),
            ConsentRule(
                selector="[id*='gdpr'], [class*='gdpr']",
                action="accept",
                text_pattern="accept|agree|allow|ok",
                timeout_ms=3000,
            ),
            # Newsletter popups
            ConsentRule(
                selector="[id*='newsletter'], [class*='newsletter']",
                action="dismiss",
                text_pattern="close|dismiss|no thanks|skip",
                timeout_ms=2000,
            ),
            # Age verification
            ConsentRule(
                selector="[id*='age'], [class*='age']",
                action="accept",
                text_pattern="yes|confirm|verify|18|21",
                timeout_ms=2000,
            ),
            # Location access
            ConsentRule(
                selector="[id*='location'], [class*='location']",
                action="accept",
                text_pattern="allow|accept|ok",
                timeout_ms=2000,
            ),
            # Generic close buttons
            ConsentRule(
                selector="[class*='close'], [class*='dismiss'], [aria-label*='close']",
                action="click",
                timeout_ms=1000,
            ),
        ]

    def get_rules_for_domain(self, domain: str) -> List[ConsentRule]:
        """
        Get consent rules specific to a domain.

        Args:
            domain: Domain name (e.g., "example.com")

        Returns:
            List of consent rules for the domain
        """
        # Domain-specific rules can be added here
        domain_rules = {
            "news.ycombinator.com": [
                ConsentRule(
                    selector=".cookie-banner", action="dismiss", timeout_ms=2000
                )
            ],
            "reddit.com": [
                ConsentRule(
                    selector="[data-testid='cookie-banner']",
                    action="accept",
                    timeout_ms=3000,
                )
            ],
        }

        rules = self.common_rules.copy()
        if domain in domain_rules:
            rules.extend(domain_rules[domain])

        return rules

    def should_handle_consent(self, url: str) -> bool:
        """
        Determine if consent handling is needed for a URL.

        Args:
            url: URL to check

        Returns:
            True if consent handling should be attempted
        """
        # Skip consent handling for certain URL patterns
        skip_patterns = [
            "api.",
            "cdn.",
            "static.",
            "assets.",
            "localhost",
            "127.0.0.1",
            "::1",
        ]

        return not any(pattern in url.lower() for pattern in skip_patterns)

    def get_consent_strategy(self, url: str) -> Dict[str, Any]:
        """
        Get consent handling strategy for a URL.

        Args:
            url: URL to handle

        Returns:
            Consent handling strategy
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        return {
            "should_handle": self.should_handle_consent(url),
            "rules": self.get_rules_for_domain(domain),
            "max_attempts": 3,
            "delay_between_attempts_ms": 1000,
        }

    def log_consent_handling(self, url: str, success: bool, details: str = ""):
        """
        Log consent handling results.

        Args:
            url: URL that was processed
            success: Whether consent handling succeeded
            details: Additional details about the process
        """
        if success:
            self.logger.debug(f"consent_resolver.py:Consent handling succeeded for {url}: {details}")
        else:
            self.logger.warning(f"consent_resolver.py:Consent handling failed for {url}: {details}")

    def get_consent_metrics(self) -> Dict[str, Any]:
        """Get consent handling metrics."""
        # This would track metrics in a real implementation
        return {
            "total_attempts": 0,
            "successful_resolutions": 0,
            "failed_resolutions": 0,
            "average_resolution_time_ms": 0,
        }
