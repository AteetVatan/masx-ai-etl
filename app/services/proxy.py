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

"""
Proxy service for managing proxy operations.

This module provides a service class for interacting with the MASX AI proxy service,
including starting proxy refresh operations and retrieving available proxies.
"""

import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from app.config import get_settings, get_service_logger
from app.core.exceptions import ServiceException


@dataclass
class ProxyStartResponse:
    """Response model for proxy start operation."""
    status: str
    duration: str
    timestamp: datetime = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ProxyListResponse:
    """Response model for proxy list operation."""
    success: bool
    data: List[str]
    message: str
    count: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        """Set count and timestamp if not provided."""
        if self.count == 0:
            self.count = len(self.data) if self.data else 0
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ProxyService:
    """
    Service class for managing proxy operations with the MASX AI proxy service.
    
    This class provides methods to:
    1. Start proxy refresh operations
    2. Retrieve available proxy lists
    3. Handle authentication and error responses
    """
    
    def __init__(self):
        """Initialize the proxy service with configuration and logging."""
        self.settings = get_settings()
        self.logger = get_service_logger("ProxyService")
        
        # Validate required configuration
        if not self.settings.proxy_api_key:
            self.logger.warning("Proxy API key not configured - proxy operations may fail")
        
        # Service endpoints
        self.base_url = self.settings.proxy_base
        self.start_endpoint = self.settings.proxy_post_start_service
        self.proxies_endpoint = self.settings.proxy_get_proxies
        
        # Headers for authentication
        self.headers = {
            "X-API-Key": self.settings.proxy_api_key,
            "Content-Type": "application/json",
            "User-Agent": "MASX-AI-ETL/1.0"
        }
        
        self.logger.info(f"ProxyService initialized with base URL: {self.base_url}")
    
    async def ping_start_proxy(self) -> ProxyStartResponse:
        """
        Start proxy refresh operation by calling the proxy service.
        
        Returns:
            ProxyStartResponse: Response containing status and duration
            
        Raises:
            ServiceException: If the proxy start operation fails
        """
        try:
            self.logger.info("Starting proxy refresh operation...")
            
            # Validate configuration
            if not self.settings.proxy_api_key:
                raise ServiceException("Proxy API key not configured")
            
            # Prepare request
            url = f"{self.base_url}{self.start_endpoint}"
            self.logger.debug(f"Calling proxy start endpoint: {url}")
            
            # Make POST request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        self.logger.info("Proxy refresh operation started successfully")
                        
                        # Parse response
                        proxy_response = ProxyStartResponse(
                            status=response_data.get("status", "unknown"),
                            duration=response_data.get("duration", "unknown")
                        )
                        
                        self.logger.info(
                            f"Proxy refresh status: {proxy_response.status}, "
                            f"Duration: {proxy_response.duration}"
                        )
                        
                        return proxy_response
                    
                    elif response.status == 401:
                        error_msg = "Unauthorized - Invalid proxy API key"
                        self.logger.error(error_msg)
                        raise ServiceException(error_msg)
                    
                    elif response.status == 429:
                        error_msg = "Rate limited - Too many requests to proxy service"
                        self.logger.warning(error_msg)
                        raise ServiceException(error_msg)
                    
                    else:
                        error_msg = f"Proxy start failed with status {response.status}"
                        try:
                            error_data = await response.text()
                            error_msg += f": {error_data}"
                        except:
                            pass
                        
                        self.logger.error(error_msg)
                        raise ServiceException(error_msg)
                        
        except aiohttp.ClientError as e:
            error_msg = f"Network error during proxy start operation: {str(e)}"
            self.logger.error(error_msg)
            raise ServiceException(error_msg)
            
        except asyncio.TimeoutError:
            error_msg = "Timeout error during proxy start operation"
            self.logger.error(error_msg)
            raise ServiceException(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error during proxy start operation: {str(e)}"
            self.logger.error(error_msg)
            raise ServiceException(error_msg)
    
    async def get_proxies(self) -> List[str]:
        """
        Retrieve list of available proxies from the proxy service.
        
        Returns:
            List[str]: List of proxy addresses in format "IP:PORT"
            
        Raises:
            ServiceException: If the proxy retrieval operation fails
        """
        try:
            self.logger.info("Retrieving proxy list...")
            
            # Validate configuration
            if not self.settings.proxy_api_key:
                raise ServiceException("Proxy API key not configured")
            
            # Prepare request
            url = f"{self.base_url}{self.proxies_endpoint}"
            self.logger.debug(f"Calling proxy list endpoint: {url}")
            
            # Make GET request
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        self.logger.info("Proxy list retrieved successfully")
                        
                        # Parse response
                        proxy_response = ProxyListResponse(
                            success=response_data.get("success", False),
                            data=response_data.get("data", []),
                            message=response_data.get("message", "")
                        )
                        
                        if not proxy_response.success:
                            error_msg = f"Proxy service returned error: {proxy_response.message}"
                            self.logger.error(error_msg)
                            raise ServiceException(error_msg)
                        
                        proxy_count = len(proxy_response.data)
                        self.logger.info(
                            f"Retrieved {proxy_count} proxies: {proxy_response.message}"
                        )
                        
                        # Validate proxy data
                        if not proxy_response.data:
                            self.logger.warning("No proxies returned from service")
                            return []
                        
                        # Return proxy list
                        return proxy_response.data
                    
                    elif response.status == 401:
                        error_msg = "Unauthorized - Invalid proxy API key"
                        self.logger.error(error_msg)
                        raise ServiceException(error_msg)
                    
                    elif response.status == 429:
                        error_msg = "Rate limited - Too many requests to proxy service"
                        self.logger.warning(error_msg)
                        raise ServiceException(error_msg)
                    
                    else:
                        error_msg = f"Proxy retrieval failed with status {response.status}"
                        try:
                            error_data = await response.text()
                            error_msg += f": {error_data}"
                        except:
                            pass
                        
                        self.logger.error(error_msg)
                        raise ServiceException(error_msg)
                        
        except aiohttp.ClientError as e:
            error_msg = f"Network error during proxy retrieval: {str(e)}"
            self.logger.error(error_msg)
            raise ServiceException(error_msg)
            
        except asyncio.TimeoutError:
            error_msg = "Timeout error during proxy retrieval"
            self.logger.error(error_msg)
            raise ServiceException(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error during proxy retrieval: {str(e)}"
            self.logger.error(error_msg)
            raise ServiceException(error_msg)
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the proxy service.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        try:
            self.logger.info("Performing proxy service health check...")
            
            # Try to get proxies as a health check
            proxies = await self.get_proxies()
            
            if proxies and len(proxies) > 0:
                self.logger.info("Proxy service health check passed")
                return True
            else:
                self.logger.warning("Proxy service health check failed - no proxies returned")
                return False
                
        except Exception as e:
            self.logger.error(f"Proxy service health check failed: {str(e)}")
            return False
    
    def get_proxy_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the proxy service configuration.
        
        Returns:
            Dict[str, Any]: Configuration statistics
        """
        return {
            "base_url": self.base_url,
            "start_endpoint": self.start_endpoint,
            "proxies_endpoint": self.proxies_endpoint,
            "api_key_configured": bool(self.settings.proxy_api_key),
            "headers_configured": bool(self.headers.get("X-API-Key")),
            "service_ready": bool(self.settings.proxy_api_key)
        }


# Convenience function for getting proxy service instance
def get_proxy_service() -> ProxyService:
    """
    Get a configured proxy service instance.
    
    Returns:
        ProxyService: Configured proxy service instance
    """
    return ProxyService()


# Example usage and testing
async def test_proxy_service():
    """Test function for the proxy service."""
    try:
        proxy_service = ProxyService()
        
        # Test health check
        health = await proxy_service.health_check()
        print(f"Health check: {health}")
        
        # Test get proxies
        proxies = await proxy_service.get_proxies()
        print(f"Retrieved {len(proxies)} proxies")
        
        # Test start proxy (if needed)
        # start_response = await proxy_service.ping_start_proxy()
        # print(f"Proxy start: {start_response}")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_proxy_service())
