from etl import DAGContext
from enums import DagContextEnum
from schemas import NewsArticle
from utils import Translator
from models import NLPBart


class Summarizer:
    """
    Summarizes raw article texts using a BART model (facebook/bart-large-cnn).
    For the prod env:
    torch==2.3.0+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html
    transformers
    (Replace cu118 with cu121 or cu117 if needed.)
    For the dev env:
    torch==2.3.0+cpu
    transformers
    """

    def __init__(self, context: DAGContext):
        self.context = context
        self.news_articles = [
            NewsArticle(**article)
            for article in self.context.pull(
                DagContextEnum.NEWS_ARTICLE_WITH_DESC.value
            )
        ]
        self.translator = Translator()
        self.nlp_bart = NLPBart(
            model_name="facebook/bart-large-cnn", model_max_tokens=1024
        )

    def summarize_all_articles(self):
        """
        Translate, compress if needed, and summarize each article.
        """
        for article in self.news_articles:
            # Step 1: Translate non-English articles to English
            article.raw_text = self.translator.translate_to_english(article.raw_text)

            # Step 2: Check if text fits the model, else compress using TF-IDF
            if not self.nlp_bart.text_suitable_for_model(
                article.raw_text, prompt_prefix="summarize: "
            ):
                article.raw_text = self.nlp_bart.compress_text_tfidf(
                    article.raw_text, prompt_prefix="summarize: "
                )

            # Step 3: Summarize using the BART model
            article.summary = self.nlp_bart.summarize_text(article.raw_text)

        # Step 4: Push serializable version of all NewsArticle objects
        serialized = [a.model_dump() for a in self.news_articles]
        self.context.push(DagContextEnum.NEWS_ARTICLE_WITH_SUMMARY.value, serialized)
