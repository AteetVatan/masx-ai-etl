"""
This class is used to summarize the text using the BART model.
"""

import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class NLPBart:
    """
    This class is used to summarize the text using the BART model.
    """

    def __init__(self, model_name="facebook/bart-large-cnn", model_max_tokens=1024):
        self.model_name = model_name
        self.model_max_tokens = model_max_tokens
        self._init_model()
        self.__init_nltk_punkt_data()

    def __init_nltk_punkt_data(self):
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt")
            nltk.download("punkt_tab")

    def __init_model(self):
        # Load model and tokenizer
        # model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.model = model.to(self.device)

    def text_suitable_for_model(self, text, prompt_prefix=""):
        """
        Check if the text token count is suitable for the model.
        """
        # prompt_prefix = "summarize: "
        # First check if it's already short enough
        token_count = len(self.tokenizer.tokenize(prompt_prefix + text))
        if token_count <= self.model_max_tokens:
            return True
        else:
            return False

    def compress_text_tfidf(self, text, prompt_prefix=""):
        """Method to shorten the text to model max token capacity by applying tfidf.
        TF-IDF (Term Frequency–Inverse Document Frequency)
        It is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.
        TF-IDF evaluates how important a word is in a document relative to a collection (corpus).
        It reduces the weight of common words (like “the”, “is”, etc.) and highlights meaningful terms.
        ***********************
        A word that appears frequently in a given chunk (high Term Frequency),
        but is rare across other chunks (high Inverse Document Frequency),
        will have a high TF-IDF score.

        """
        sentences = self.__safe_sent_tokenize(text)
        if not sentences:
            print("TF-IDF compression failed due to empty/faulty text.")
            return None

        # prompt_prefix = "summarize: "
        # Step 1: Apply TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        scores = X.sum(axis=1).A1  # Flatten sparse matrix to 1D array

        # Pair sentences with scores but preserve original order
        sentence_score_pairs = [(sent, score) for sent, score in zip(sentences, scores)]

        # Optional: Remove low-score noise
        score_threshold = max(scores) * 0.2  # keep top 20% importance
        sentence_score_pairs = [
            (sent, score)
            for sent, score in sentence_score_pairs
            if score >= score_threshold
        ]

        # Step 3: Accumulate sentences in original order until token limit
        selected = []
        for sent, _ in sentence_score_pairs:
            test_input = prompt_prefix + " ".join(selected + [sent])
            if len(self.tokenizer.tokenize(test_input)) > self.model_max_tokens:
                break
            selected.append(sent)

        compressed = " ".join(selected)
        return compressed

    def __safe_sent_tokenize(self, text, lang="english"):
        """
        Safe sentence tokenization using nltk.
        """
        try:
            sentences = sent_tokenize(text, language=lang)
        except LookupError:
            print(f"No tokenizer model for {lang}. Falling back to English.")
            sentences = sent_tokenize(text, language="english")
        except Exception as e:
            print(f"Tokenization failed: {e}")
            sentences = None

        if not sentences:
            # nltk tokenize failed, so we return the text as is
            # Fallback to naive splitting
            sentences = text.split(".")
            # heuristic to filter out very short or irrelevant sentences  < 20 chars
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            if not sentences:
                print("TF-IDF compression failed due to empty/faulty text.")
                return None  # Final fallback
        return sentences

    def summarize_text(self, text: str):
        """
        Summarize the text using the BART model.
        """
        # Step 1: Tokenize for BART
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=self.model_max_tokens, truncation=True
        )

        input_ids = inputs["input_ids"].to(self.device)

        # Step 2: Generate summary
        summary_ids = self.model.generate(
            input_ids,
            max_length=256,
            num_beams=4,
            length_penalty=1.5,
            early_stopping=True,
        )

        # Decode and store result
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
