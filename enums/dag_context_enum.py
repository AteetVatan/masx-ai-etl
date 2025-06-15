from enum import Enum


class DagContextEnum(Enum):
    VALID_PROXIES = "valid_proxies"
    NEWS_ARTICLES = "news_articles"
    NEWS_ARTICLE_WITH_DESC = "news_article_with_desc"
    NEWS_ARTICLE_WITH_SUMMARY = "news_article_with_summary"
    NEWS_ARTICLE_WITH_KEYWORDS = "news_article_with_keywords"
    ENV_CONFIG = "env_config"
