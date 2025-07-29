from nlp import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from singleton import ModelManager
from nlp import Translator, NLPUtils
from config import get_service_logger, get_settings
from etl_data.etl_models import FeedModel
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from core.exceptions import ServiceException

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

    def __init__(self, feeds: list[FeedModel]):
        self.feeds = feeds
        self.summarization_model, self.summarization_tokenizer, self.device = (
            ModelManager.get_summarization_model()
        )
        self.translator = Translator()
        self.logger = get_service_logger("Summarizer")
        self.settings = get_settings()
        self.max_workers = self.settings.max_workers
        self.prompt_prefix = "summarize: "

    def summarize_all_feeds(self):
        """
        Translate, compress if needed, and summarize each article.
        """       
        try:
            summarized_feeds = []
            # apply the max_workers for threading
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self.__summarize_feed, feed)
                    for feed in self.feeds
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        summarized_feeds.append(result)
            return summarized_feeds
        except Exception as e:
            self.logger.error(f"[Summarizer] Error summarizing feeds: {e}")
            raise ServiceException(f"Error summarizing feeds: {e}")

    def __summarize_feed(self, feed: FeedModel):
        """
        Summarize a single feed.
        """ 
        try:
            
            # Step 1: Translate non-English articles to English    
            try:                   
                feed.raw_text_en = self.translator.ensure_english(feed.raw_text)
            except Exception as e:
                self.logger.error(f"[Summarizer] Error translating feed: {e}")
                raise ServiceException(f"Error translating feed: {e}")

            # Step 2: Check if text fits the model, else compress using TF-IDF
            try:
                #tokenizer = self.summarization_tokenizer
                tokenizer = self.summarization_tokenizer.__class__.from_pretrained(
                    self.summarization_tokenizer.name_or_path
                )
                max_tokens = ModelManager.get_summarization_model_max_tokens()
                if not NLPUtils.text_suitable_for_model(
                    tokenizer,
                    feed.raw_text_en,
                    max_tokens,
                ):
                    self.logger.info(f"[Summarizer] Compressing text using TF-IDF")
                    text = NLPUtils.compress_text_tfidf(
                        tokenizer,
                        feed.raw_text_en,
                        max_tokens,
                        prompt_prefix=self.prompt_prefix,
                    )
                else:
                        text = feed.raw_text_en
            except Exception as e:
                self.logger.error(f"[Summarizer] Error compressing text: {e}")
                raise ServiceException(f"Error compressing text: {e}")

            # Step 3: Summarize using the BART model
            try:
                self.logger.info(f"[Summarizer] Summarizing text using BART model")
                max_tokens = ModelManager.get_summarization_model_max_tokens()
                summarizer = self.summarization_model
                tokenizer = self.summarization_tokenizer.__class__.from_pretrained(
                    self.summarization_tokenizer.name_or_path
                )
                device = self.device
                feed.summary = NLPUtils.summarize_text(
                    summarizer,
                    tokenizer,
                    device,
                    self.prompt_prefix + text,
                    max_tokens
                    )
            except Exception as e:
                self.logger.error(f"[Summarizer] Error summarizing text: {e}")
                raise ServiceException(f"Error summarizing text: {e}")

            # step 5: Generate questions from summary
            # self.generate_questions_from_summary(max_questions=3)

            # Step 4: Push serializable version of all NewsArticle objects
            #serialized = [a.model_dump() for a in self.news_articles]
            return feed
        except Exception as e:
            self.logger.error(f"[Summarizer] Error summarizing feed: {e}")
            raise ServiceException(f"Error summarizing feed: {e}")

    # ─── Generate Questions from Summary ──────────────────────────────────────────────On trial for now
    def generate_questions_from_summary(self, max_questions: int = 3) -> list:
        """
        Generate up to 3 concise questions from a given summary using Flan-T5-Large.
        """

        # Load the tokenizer and model from Hugging Face Hub
        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        for feed in self.feeds:

            prompt = f"Generate {max_questions} questions based on this summary:\n{feed.summary}"

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(model.device)

            # Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Split into lines, return up to `max_questions` questions
            questions = [
                q.strip("- ").strip() for q in generated_text.split("\n") if q.strip()
            ]
            feed.questions = questions