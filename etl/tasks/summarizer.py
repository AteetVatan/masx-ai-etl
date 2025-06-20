from schemas import NewsArticle
from utils import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from singleton import ModelManager
from utils import Translator, NLPUtils


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

    def __init__(self, news_articles: list[NewsArticle]):
        self.news_articles = news_articles
        self.bart_model, self.bart_tokenizer, self.device = (
            ModelManager.get_bart_model()
        )
        self.translator = Translator()

    def summarize_all_articles(self):
        """
        Translate, compress if needed, and summarize each article.
        """
        prompt_prefix = "summarize: "

        for article in self.news_articles:
            # Step 1: Translate non-English articles to English
            article.raw_text = self.translator.ensure_english(article.raw_text)

            # Step 2: Check if text fits the model, else compress using TF-IDF
            if not NLPUtils.text_suitable_for_model(
                self.bart_tokenizer,
                article.raw_text,
                ModelManager.get_model_max_tokens(),
            ):

                text = NLPUtils.compress_text_tfidf(
                    self.bart_tokenizer,
                    article.raw_text,
                    ModelManager.get_model_max_tokens(),
                    prompt_prefix=prompt_prefix,
                )
            else:
                text = article.raw_text

            # Step 3: Summarize using the BART model
            article.summary = NLPUtils.summarize_text(
                self.bart_model,
                self.bart_tokenizer,
                self.device,
                prompt_prefix + text,
                ModelManager.get_model_max_tokens()
            )

        # step 5: Generate questions from summary
        # self.generate_questions_from_summary(max_questions=3)

        # Step 4: Push serializable version of all NewsArticle objects
        #serialized = [a.model_dump() for a in self.news_articles]
        return self.news_articles

    # ─── Generate Questions from Summary ──────────────────────────────────────────────On trial for now
    def generate_questions_from_summary(self, max_questions: int = 3) -> list:
        """
        Generate up to 3 concise questions from a given summary using Flan-T5-Large.
        """

        # Load the tokenizer and model from Hugging Face Hub
        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        for article in self.news_articles:

            prompt = f"Generate {max_questions} questions based on this summary:\n{article.summary}"

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
            article.questions = questions