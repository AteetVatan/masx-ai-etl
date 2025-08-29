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
This module handles all BeautifulSoup-related operations in the MASX AI News ETL pipeline.
"""

import html
from random import choice

import requests
from bs4 import BeautifulSoup


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
            response = requests.get(
                url, headers=headers, timeout=3600
            )  # 1 hour for complex pages
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
