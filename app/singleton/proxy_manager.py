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
This module handles all proxy-related operations in the MASX AI News ETL pipeline.
"""

import random
from datetime import datetime, timedelta
import asyncio
import requests
from bs4 import BeautifulSoup

from app.config import headers_list, get_settings, get_service_logger
from app.core.concurrency import CPUExecutors


# convert this class to a singleton
class ProxyManager:
    """
    Handles all proxy-related operations in the MASX AI News ETL pipeline.
    """

    __proxies = []
    __settings = get_settings()
    __proxy_webpage = __settings.proxy_webpage
    __proxy_testing_url = __settings.proxy_testing_url
    __headers_list = headers_list
    __proxy_expiration = timedelta(minutes=5)
    __proxy_timestamp = datetime.now()
    __logger = get_service_logger("ProxyManager")

    # Initialize CPU executors for async processing
    __cpu_executors = CPUExecutors()

    @classmethod
    def proxies(cls):
        """
        Get available proxies (synchronous wrapper for backward compatibility).
        """
        # Use asyncio.run for synchronous access
        return asyncio.run(cls.proxies_async())

    @classmethod
    async def proxies_async(cls):
        """
        Get available proxies asynchronously.
        """
        if cls.__proxies:
            if cls.__proxy_timestamp + cls.__proxy_expiration < datetime.now():
                cls.__proxies = []

        if not cls.__proxies:
            # get all proxies
            cls.__logger.info("proxy_manager.py:Getting proxies from proxy site...")
            all_proxies = cls.__get_proxies()
            # test proxies
            cls.__logger.info("proxy_manager.py:Testing proxies...")
            proxies = list(set(await cls.__test_proxy(all_proxies)))
            cls.__proxies = proxies
            cls.__logger.info(f"proxy_manager.py:Found {len(cls.__proxies)} proxies")
            cls.__proxy_timestamp = datetime.now()

        return cls.__proxies

    @classmethod
    def __get_proxies_2(cls):

        import requests
        import json

        # URL of the raw JSON file from proxifly using jsDelivr CDN
        url = "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/all/data.json"

        try:
            # Download the JSON file
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes
            proxies = response.json()

            # Extract only the IPs
            ip_list = [
                str(proxy["ip"]) + ":" + str(proxy["port"])
                for proxy in proxies
                if "ip" in proxy
            ]

        except requests.exceptions.RequestException as e:
            print(f"Error downloading proxies: {e}")

        return ip_list

    @classmethod
    def __get_proxies(cls):
        """
        Get a list of proxies from a proxy site).
        """
        proxies = []
        headers = random.choice(cls.__headers_list)
        page = requests.get(cls.__proxy_webpage, headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")
        for row in soup.find("tbody").find_all("tr"):
            proxy = row.find_all("td")[0].text + ":" + row.find_all("td")[1].text
            proxies.append(proxy)

        return proxies

    @classmethod
    async def __test_proxy(cls, proxies):
        """Checks which ones actually work using async CPU executors."""
        try:
            # Process proxies in batches for efficiency
            batch_size = 20
            valid_proxies = []

            for i in range(0, len(proxies), batch_size):
                batch = proxies[i : i + batch_size]
                batch_results = await cls._test_proxy_batch(batch)
                valid_proxies.extend(batch_results)

            return valid_proxies

        except Exception as e:
            cls.__logger.error(f"proxy_manager.py:Proxy testing failed: {e}")
            # Fallback to synchronous testing
            return cls._test_proxy_sync(proxies)

    @classmethod
    async def _test_proxy_batch(cls, proxies):
        """Test a batch of proxies concurrently."""
        try:
            # Create tasks for concurrent testing
            tasks = [
                cls.__cpu_executors.run_in_thread(cls.__test_single_proxy, proxy)
                for proxy in proxies
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            valid_proxies = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    cls.__logger.debug(f"proxy_manager.py:Proxy {proxies[i]} testing failed: {result}")
                    continue

                if result:
                    valid_proxies.append(proxies[i])

            return valid_proxies

        except Exception as e:
            cls.__logger.error(f"proxy_manager.py:Batch proxy testing failed: {e}")
            return []

    @classmethod
    def _test_proxy_sync(cls, proxies):
        """Synchronous fallback for proxy testing."""
        valid_proxies = []
        for proxy in proxies:
            try:
                if cls.__test_single_proxy(proxy):
                    valid_proxies.append(proxy)
            except Exception as e:
                cls.__logger.debug(f"proxy_manager.py:Proxy {proxy} testing failed: {e}")
        return valid_proxies

    @classmethod
    def __test_single_proxy(cls, proxy):
        """Test a single proxy"""
        headers = random.choice(cls.__headers_list)
        try:
            resp = requests.get(
                cls.__proxy_testing_url,
                headers=headers,
                proxies={"http": proxy, "https": proxy},
                timeout=3,
            )
            if resp.status_code == 200:
                return True
        except:
            pass
        return False
