# """
# This module handles all news-related operations in the MASX AI News ETL pipeline.
# """

# import requests
# from enums import EnvKeyEnum
# from schemas import ArticleListSchema, NewsArticle
# from singleton import EnvManager
# #from mock_data.mock_data import mock_data


# class NewsManager:
#     """
#     This class handles all news-related operations in the MASX AI News ETL pipeline.
#     """

#     def __init__(self):
#         self.env_config = EnvManager.get_env_vars()
#         self.api_key = self.env_config[EnvKeyEnum.MASX_GDELT_API_KEY.value]
#         self.api_url = self.env_config[EnvKeyEnum.MASX_GDELT_API_URL.value]
#         self.keywords = self.env_config[EnvKeyEnum.MASX_GDELT_API_KEYWORDS.value]
#         self.max_records = self.env_config[EnvKeyEnum.MASX_GDELT_MAX_RECORDS.value]

#     def news_articles(self) -> list[NewsArticle]:
#         """
#         Get the news articles from the GDELT API.
#         """
#         if not self.env_config[EnvKeyEnum.DEBUG.value]:
#             articles = self.__fetch_gdelt_articles()
#         else:
#             articles = [] # mock_data # list[dict}]

#         gdelt_articles = self.__validate_gdelt_articles(articles)

#         if gdelt_articles:
#             articles = ArticleListSchema.model_validate(gdelt_articles)
#             return self.__map_to_news_articles_schema(articles)
#         raise ValueError("Invalid GDELT articles")

#     def __fetch_gdelt_articles(self):
#         """
#         Fetch the GDELT articles from the API.
#         """
#         try:
#             response = requests.post(
#                 self.api_url,
#                 headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
#                 json={"keyword": self.keywords, "maxrecords": int(self.max_records)},
#             )
#             response.raise_for_status()
#             return response.json()
#         except requests.exceptions.RequestException as err:
#             print(f"Request failed: {err}")
#             return []

#     def __validate_gdelt_articles(self, gdelt_articles: list):
#         """
#         Pydantic model validation.
#         Validate the GDELT articles using ArticleListSchema.
#         Returns True if valid, False otherwise.
#         """
#         try:
#             validated = ArticleListSchema.model_validate(gdelt_articles)
#             return validated.root
#         except Exception as e:
#             print(f"[SCHEMA ERROR] Validation failed: {e}")
#             return []

#     def __map_to_news_articles_schema(
#         self, articles: ArticleListSchema
#     ) -> list[NewsArticle]:
#         """
#         Map the GDELT articles to the NewsArticle schema.
#         """
#         news_articles = []
#         for article in articles.root:  # for article in articles.root:
#             news_articles.append(
#                 NewsArticle(
#                     url=article.url,
#                     title=article.title,
#                     raw_text="",
#                     summary="",
#                     socialimage=article.socialimage,
#                     domain=article.domain,
#                     language=article.language,
#                     sourcecountry=article.sourcecountry,
#                 )
#             )
#         return news_articles
