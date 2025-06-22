from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import torch

class NLPUtils:

    @staticmethod
    def text_suitable_for_model(
        bart_tokenizer: AutoTokenizer, text: str, model_max_tokens: int
    ):
        """
        Check if the text token count is suitable for the model.
        """
        # prompt_prefix = "summarize: "
        # First check if it's already short enough
        token_count = len(bart_tokenizer.tokenize(text))
        if token_count <= model_max_tokens:
            return True
        else:
            return False

    @staticmethod
    def compress_text_tfidf(
        bart_tokenizer: AutoTokenizer,
        text: str,
        model_max_tokens: int,
        prompt_prefix: str = "summarize: ",
    ):
        """Method to shorten the text to model max token capacity by applying tfidf.
        TF-IDF (Term Frequency–Inverse Document Frequency)
        It is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.
        TF-IDF evaluates how important a word is in a document relative to a collection (corpus).
        It reduces the weight of common words (like "the", "is", etc.) and highlights meaningful terms.
        ***********************
        A word that appears frequently in a given chunk (high Term Frequency),
        but is rare across other chunks (high Inverse Document Frequency),
        will have a high TF-IDF score.

        """
        sentences = NLPUtils.safe_sent_tokenize(text)
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
            if len(bart_tokenizer.tokenize(test_input)) > model_max_tokens:
                break
            selected.append(sent)

        compressed = " ".join(selected)
        return compressed

    @staticmethod
    def safe_sent_tokenize(text, lang="english"):
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

    @staticmethod
    def summarize_text(
        bart_model: AutoModelForSeq2SeqLM,
        bart_tokenizer: AutoTokenizer,
        device: torch.device,
        text: str,
        model_max_tokens: int,
    ):
        """
        Summarize the text using the BART model.
        """
        # Step 1: Tokenize for BART
        inputs = bart_tokenizer(
            text, return_tensors="pt", max_length=model_max_tokens, truncation=True
        )

        input_ids = inputs["input_ids"].to(device)

        # Step 2: Generate summary
        summary_ids = bart_model.generate(
            input_ids,
            max_length=256,
            num_beams=4,
            length_penalty=1.5,
            early_stopping=True,
        )

        # Decode and store result
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
