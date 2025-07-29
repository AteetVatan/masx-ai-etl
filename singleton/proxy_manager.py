"""
This module handles all proxy-related operations in the MASX AI News ETL pipeline.
"""

from concurrent.futures import ThreadPoolExecutor
import random
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
from config import get_settings, headers_list, get_service_logger




# convert this class to a singleton
class ProxyManager:
    """
    Handles all proxy-related operations in the MASX AI News ETL pipeline.
    """

    __proxies = []
    __settings = get_settings()
    __proxy_webpage = __settings.proxy_webpage
    __proxy_testing_url = __settings.proxy_testing_url
    __max_workers = __settings.max_workers
    __headers_list = headers_list
    __proxy_expiration = timedelta(minutes=5)
    __proxy_timestamp = datetime.now()
    __logger = get_service_logger("ProxyManager")

    @classmethod
    def proxies(cls):
        """
        Get avaliable proxies
        (either from instance or fresh from a proxy site)
        """
        if cls.__proxies:
            if cls.__proxy_timestamp + cls.__proxy_expiration < datetime.now():
                cls.__proxies = []

        if not cls.__proxies:
            # get all proxies
            all_proxies = cls.__get_proxies()
            # test proxies
            cls.__proxies = list(set(cls.__test_proxy(all_proxies)))
            cls.__proxy_timestamp = datetime.now()

        return cls.__proxies

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
    def __test_proxy(cls, proxies):
        """Checks which ones actually work."""
        with ThreadPoolExecutor(max_workers=cls.__max_workers) as executor:
            results = list(executor.map(cls.__test_single_proxy, proxies))
        return (proxy for valid, proxy in zip(results, proxies) if valid)

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
