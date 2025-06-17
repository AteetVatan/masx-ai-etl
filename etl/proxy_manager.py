"""
This module handles all proxy-related operations in the MASX AI News ETL pipeline.
"""

from concurrent.futures import ThreadPoolExecutor
import random
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
import concurrent.futures
from etl.config import headers_list
from etl import DAGContext
from enums import DagContextEnum, EnvKeyEnum

logger = logging.getLogger("proxy_manager")


class ProxyManager:
    """
    Handles all proxy-related operations in the MASX AI News ETL pipeline.
    Uses DAGContextUtils (XComs) for DAG-run-scoped communication.
    """

    def __init__(self, context: DAGContext):
        self.context = context
        self.proxies = []
        self.env_config = self.context.pull(DagContextEnum.ENV_CONFIG.value)
        self.proxy_webpage = self.env_config[EnvKeyEnum.PROXY_WEBPAGE.value]
        self.proxy_testing_url = self.env_config[EnvKeyEnum.PROXY_TESTING_URL.value]
        self.max_workers = self.env_config[EnvKeyEnum.MAX_WORKERS.value]
        self.headers_list = headers_list
        self.proxy_expiration = timedelta(minutes=5)
        self.proxy_timestamp = datetime.now()
        self.proxies = []

    def get_valid_proxies(self):
        """
        Get valid proxies from the proxy manager.
        """
        proxies = self.__get_proxies()
        valid_proxies = list(self.__test_proxy(proxies))
        return valid_proxies

    def __get_proxies(self):
        """
        Get a list of proxies (either from Redis cache or fresh from a proxy site).
        """
        if self.proxies:
            if self.proxy_timestamp + self.proxy_expiration < datetime.now():
                self.proxies = []

        if not self.proxies:
            headers = random.choice(self.headers_list)
            page = requests.get(self.proxy_webpage, headers=headers)
            soup = BeautifulSoup(page.content, "html.parser")
            for row in soup.find("tbody").find_all("tr"):
                proxy = row.find_all("td")[0].text + ":" + row.find_all("td")[1].text
                self.proxies.append(proxy)
            self.proxy_timestamp = datetime.now()

        return self.proxies

    def __test_proxy(self, proxies):
        """Checks which ones actually work."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.__test_single_proxy, proxies))
        return (proxy for valid, proxy in zip(results, proxies) if valid)

    def __test_single_proxy(self, proxy):
        """Test a single proxy"""
        headers = random.choice(self.headers_list)
        try:
            resp = requests.get(
                self.proxy_testing_url,
                headers=headers,
                proxies={"http": proxy, "https": proxy},
                timeout=3,
            )
            if resp.status_code == 200:
                return True
        except:
            pass
        return False
