"""
This module handles all BeautifulSoup-related operations in the MASX AI News ETL pipeline.
"""

import requests
from bs4 import BeautifulSoup
from random import choice
import html


class BeautifulSoupExtractor:
    """
    This class handles all BeautifulSoup-related operations in the MASX AI News ETL pipeline.
    """

    @staticmethod
    def beautiful_soup_scrape(url, proxy):
        """
        Scrape the article using BeautifulSoup.
        """
        try:
            headers = {
                "User-Agent": proxy if isinstance(proxy, str) else choice(list(proxy))
            }
            # url = "https://baijiahao.baidu.com/s?id=1834495451984756877"
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = response.apparent_encoding
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            print(f"⚠️ Failed to scrape {url}: {e}")
            return None

    @staticmethod
    def extract_text_from_soup(soup):
        """
        Extract the text from the BeautifulSoup object.
        """
        try:
            # title_tag = soup.find('title')
            # title = title_tag.get_text(strip=True) if title_tag else "Untitled"
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs)
            text = html.unescape(text.replace("\xa0", " "))
            return text
        except Exception as e:
            print(f"Failed to extract text from {soup}: {e}")
            return None
