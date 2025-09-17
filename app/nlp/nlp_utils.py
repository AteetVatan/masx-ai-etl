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

"""This module contains utility functions for NLP operations."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import math
import numpy as np
from numpy import array as np_array
import unicodedata
import logging

logger = logging.getLogger(__name__)


def load_spacy_model(name: str, blank_fallback: str):
    try:
        nlp = spacy.load(name)
        logger.info(f"NLPUtils: Successfully loaded {name}")
        return nlp
    except OSError as e:  # model package not found
        logger.warning(
            f"NLPUtils: spaCy model {name} not installed. Falling back to blank {blank_fallback}. Error: {e}"
        )
        nlp = spacy.blank(blank_fallback)
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except Exception as e:
        logger.error(f"NLPUtils: Unexpected error loading {name}: {e}")
        raise  # don't silently fallback on unrelated errors!


class NLPUtils:
    """Utility functions for NLP operations."""

    logger = logging.getLogger(__name__)

    nlp_en = load_spacy_model("en_core_web_sm", "en")
    nlp_all = load_spacy_model("xx_ent_wiki_sm", "xx")

    NLTK_LANGS = {
        "en": "english",
        "fr": "french",
        "de": "german",
        "es": "spanish",
        "pt": "portuguese",
        "it": "italian",
        "nl": "dutch",
        "tr": "turkish",
        "sv": "swedish",
        "da": "danish",
        "fi": "finnish",
        "no": "norwegian",
        "pl": "polish",
        "sl": "slovene",
        "et": "estonian",
        "el": "greek",
        "ru": "russian",
    }

    @staticmethod
    def load_spacy_model(name: str, blank_fallback: str):
        try:
            nlp = spacy.load(name)
            NLPUtils.logger.info(f"NLPUtils: Successfully loaded {name}")
            return nlp
        except OSError as e:  # model package not found
            NLPUtils.logger.warning(
                f"NLPUtils: spaCy model {name} not installed. Falling back to blank {blank_fallback}. Error: {e}"
            )
            nlp = spacy.blank(blank_fallback)
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception as e:
            NLPUtils.logger.error(f"NLPUtils: Unexpected error loading {name}: {e}")
            raise  # don't silently fallback on unrelated errors!

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
        scores_percentage: float = 0.2,
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
        try:
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
            sentence_score_pairs = [
                (sent, score) for sent, score in zip(sentences, scores)
            ]

            # Optional: Remove low-score noise
            score_threshold = max(scores) * scores_percentage  # keep top 20% importance
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
        except Exception as e:
            raise e

    @staticmethod
    def compress_text_tfidf_with_fraction(
        bart_tokenizer: AutoTokenizer,
        text: str,
        model_max_tokens: int,
        prompt_prefix: str = "summarize: ",
        keep_fraction: float = 0.3,  # keep top 30% sentences by score (adjust 0.2–0.4)
    ):
        import re
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        sentences = NLPUtils.safe_sent_tokenize(text)
        if not sentences:
            return ""

        # TF-IDF over sentences
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, lowercase=True
        )
        X = vectorizer.fit_transform(sentences)  # [n_sent, vocab]
        scores = np.asarray(X.sum(axis=1)).ravel()  # simple salience

        # Light factual boost (numbers/dates)
        has_num = np.array([bool(re.search(r"\d", s)) for s in sentences], dtype=float)
        scores = scores + 0.1 * has_num

        # Select top-K by rank (not threshold)
        n = len(sentences)
        k = max(1, int(n * keep_fraction))
        top_idx = np.argsort(-scores)[:k]

        # Optional tiny redundancy control (MMR-lite)
        # Re-rank top_idx to reduce duplicates
        from sklearn.metrics.pairwise import cosine_similarity

        sel = []
        cand = list(top_idx)
        sims_mat = cosine_similarity(X[cand], X[cand]) if len(cand) > 1 else None
        lam = 0.7
        while cand:
            if not sel:
                sel.append(cand.pop(0))
                continue
            # compute max similarity to already selected
            max_sims = []
            for i, ci in enumerate(cand):
                s = 0.0
                for sj in sel:
                    # cosine between sentence ci and sj
                    s = max(s, cosine_similarity(X[ci], X[sj])[0, 0])
                max_sims.append(s)
            mmr = lam * scores[cand] - (1 - lam) * np.array(max_sims)
            pick = cand[int(np.argmax(mmr))]
            sel.append(pick)
            cand.remove(pick)

        # Restore original narrative order
        sel.sort()

        # Accumulate until token budget
        out = []
        for idx in sel:
            test = prompt_prefix + (" ".join(out + [sentences[idx]]))
            if len(bart_tokenizer.tokenize(test)) > model_max_tokens:
                break
            out.append(sentences[idx])

        return " ".join(out)

    @staticmethod
    def chunk_text(
        text: str,
        max_chars: int,
        prompt_prefix: str = "summarize: ",
    ):
        """Method to chunk the text into smaller chunks."""
        try:
            max_chars = max_chars + len(prompt_prefix)
            sentences = NLPUtils.safe_sent_tokenize(text)
            if not sentences:
                print("TF-IDF compression failed due to empty/faulty text.")
                return None

            chunks = []
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    if len(sentence) > max_chars:
                        for i in range(0, len(sentence), max_chars):
                            chunks.append(sentence[i : i + max_chars])
                        current_chunk = ""
                    else:
                        current_chunk = sentence + " "
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            return chunks
        except Exception as e:
            raise e

    @staticmethod
    def safe_sent_tokenize(text: str, lang_iso: str = "en"):
        """
        Safe sentence tokenization using NLTK with proper multi-language support.
        Falls back to regex/naive splitting for unsupported langs.
        """
        try:
            if lang_iso.lower() not in NLPUtils.NLTK_LANGS:
                logger.info(f"Using language-agnostic tokenization for '{lang_iso}'")
                return NLPUtils._language_agnostic_tokenize(text, lang_iso)

            lang = NLPUtils.NLTK_LANGS[lang_iso.lower()]

            # First try with specified language
            return sent_tokenize(text, language=lang)

        except LookupError:
            logger.warning(
                f"NLTK tokenizer model for '{lang_iso}' not found. Attempting to download..."
            )
            try:
                import nltk

                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)

                lang = NLPUtils.NLTK_LANGS.get(lang_iso.lower(), "english")
                sentences = sent_tokenize(text, language=lang)
                logger.info(f"Successfully downloaded and used NLTK model for '{lang}'")
                return sentences

            except Exception as download_error:
                logger.warning(
                    f"Failed to download NLTK model for '{lang_iso}': {download_error}"
                )

                if lang_iso.lower() != "en":
                    logger.info(
                        f"Using language-agnostic tokenization for '{lang_iso}'"
                    )
                    return NLPUtils._language_agnostic_tokenize(text, lang_iso)

                # English fallback
                try:
                    sentences = sent_tokenize(text, language="english")
                    logger.info("Using English tokenizer as fallback")
                    return sentences
                except Exception as english_error:
                    logger.error(f"English tokenizer also failed: {english_error}")
                    return NLPUtils._naive_sentence_split(text)

        except Exception as e:
            logger.error(f"Tokenization failed with unexpected error: {e}")
            return NLPUtils._naive_sentence_split(text)

    @staticmethod
    def _language_agnostic_tokenize(text: str, lang: str) -> List[str]:
        """
        Language-agnostic sentence tokenization for languages without NLTK models.
        Uses regex patterns and heuristics that work across multiple languages.
        """
        import re

        # Common sentence ending patterns across languages
        sentence_endings = [
            r"[.!?]+",  # English, German, French, Spanish
            r"[。！？]+",  # Chinese, Japanese
            r"[।!?]+",  # Hindi, Bengali
            r"[؟!]+",  # Arabic
            r"[!?]+",  # Russian, other Cyrillic
            r"[.!?。！？]+",  # Korean
            r"[.!?。！？؟!।]+",  # Universal fallback
        ]

        # Combine all patterns
        pattern = "|".join(sentence_endings)

        # Split on sentence endings, preserving the endings
        sentences = re.split(f"({pattern})", text)

        # Reconstruct sentences with their endings
        result = []
        current_sentence = ""

        for i, part in enumerate(sentences):
            if re.match(pattern, part):
                # This is a sentence ending
                current_sentence += part
                if current_sentence.strip():
                    result.append(current_sentence.strip())
                current_sentence = ""
            else:
                # This is text content
                current_sentence += part

        # Add any remaining text
        if current_sentence.strip():
            result.append(current_sentence.strip())

        # Filter out very short sentences
        result = [s for s in result if len(s.strip()) > 20]

        if not result:
            logger.warning(
                f"Language-agnostic tokenization failed for '{lang}', using naive split"
            )
            return NLPUtils._naive_sentence_split(text)

        return result

    @staticmethod
    def _naive_sentence_split(text: str) -> List[str]:
        """Naive sentence splitting as final fallback."""
        sentences = text.split(".")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if not sentences:
            logger.warning("All tokenization methods failed, returning single sentence")
            return [text] if text.strip() else []
        return sentences

    @staticmethod
    def summarize_text(
        bart_model: AutoModelForSeq2SeqLM,
        bart_tokenizer: AutoTokenizer,
        device: torch.device,
        text: str,
        model_max_tokens: int,
        gen_kwargs: Optional[Dict] = None,
    ) -> str:
        """
        Summarize the text using the BART model.
        """
        inputs = bart_tokenizer(
            text, return_tensors="pt", max_length=model_max_tokens, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)

        kwargs = dict(
            max_length=min(1024, 800),
            min_length=250,
            num_beams=5,
            length_penalty=1.0,
            eos_token_id=bart_tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
        )
        if gen_kwargs:
            kwargs.update(gen_kwargs)
        summary_ids = bart_model.generate(input_ids, **kwargs)
        return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # ---------- NEW: batch summarize windows ----------
    @staticmethod
    def summarize_windows(
        bart_model: AutoModelForSeq2SeqLM,
        bart_tokenizer: AutoTokenizer,
        device: torch.device,
        windows: List[str],
        model_max_tokens: int,
        gen_kwargs: Optional[Dict] = None,
    ) -> List[str]:
        """
        Batch summarize windows using the BART model.
        """
        out = []
        for w in windows:
            out.append(
                NLPUtils.summarize_text(
                    bart_model, bart_tokenizer, device, w, model_max_tokens, gen_kwargs
                )
            )
        return out

    # ---------- NEW: merge & polish with structured prompt ----------
    @staticmethod
    def merge_and_polish(
        bart_model: AutoModelForSeq2SeqLM,
        bart_tokenizer: AutoTokenizer,
        device: torch.device,
        chunk_summaries: List[str],
        model_max_tokens: int,
        target_tokens: int = 800,
    ) -> str:
        """
        Merge and polish chunk summaries with a structured, fact-only prompt.
        Deterministic (no sampling), token-safe trimming for the encoder.
        """

        # 1) Build instruction header (short, factual, structured)
        header = (
            "summarize: Produce a strictly factual synthesis from the Sources.\n"
            "Rules:\n"
            " - Use only facts present in Sources; do NOT add new claims.\n"
            " - Preserve names, numbers, currency, and attribution exactly.\n"
            " - Use absolute dates if present; otherwise omit dates (no 'today'/'recent').\n"
            " - If a section has no content, write '—'.\n"
            "Format:\n"
            "LEDE: 1–2 sentences.\n"
            "- Key facts: actors, dates, magnitudes.\n"
            "- What changed vs. prior.\n"
            "- Uncertainties or conflicting reports.\n"
            "OUTLOOK: one sentence.\n"
            "Sources:\n"
        )

        # 2) Join sources (label them; minimal noise)
        sources = "\n\n".join(
            f"[{i+1}] {s.strip()}"
            for i, s in enumerate(chunk_summaries)
            if s and s.strip()
        )

        total_tokens = len(bart_tokenizer.tokenize(sources))
        # if total_tokens > model_max_tokens:
        #     sources = sources[:model_max_tokens]

        # 3) Token budgeting for encoder (BART ≈ 1024)
        encoder_cap = min(
            model_max_tokens, getattr(bart_tokenizer, "model_max_length", 1024) or 1024
        )

        header_ids = bart_tokenizer(
            header, add_special_tokens=True, return_tensors="pt", truncation=False
        )["input_ids"][0]

        # Whatever remains is for sources
        avail = int(encoder_cap - header_ids.shape[-1])
        if avail < 64:
            # If the header is too big (unlikely), shrink with a tiny fallback
            header = "summarize: factual synthesis. Format: LEDE; Key facts; Changes; Uncertainties; OUTLOOK.\nSources:\n"
            header_ids = bart_tokenizer(
                header, add_special_tokens=True, return_tensors="pt"
            )["input_ids"][0]
            avail = max(64, int(encoder_cap - header_ids.shape[-1]))

        # Tokenize sources with truncation to fit the remaining budget
        src = bart_tokenizer(
            sources,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=avail,
        )["input_ids"][0]

        # 4) Compose final encoder input
        input_ids = torch.cat([header_ids, src], dim=-1).unsqueeze(0).to(device)

        # 5) Deterministic generation (no sampling, greedy)
        gen_kwargs = {
            "do_sample": False,
            "num_beams": 1,  # greedy to minimize drift
            # "max_new_tokens": min(400, max(120, target_tokens)),
            "min_new_tokens": 500 if total_tokens > 1000 else 200,
            "max_new_tokens": 600 if total_tokens > 1000 else 400,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
            "eos_token_id": bart_tokenizer.eos_token_id,
            "pad_token_id": bart_tokenizer.pad_token_id,
        }

        with torch.no_grad():
            output_ids = bart_model.generate(input_ids, **gen_kwargs)

        return bart_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    @staticmethod
    def merge_and_polish_1(
        bart_model: AutoModelForSeq2SeqLM,
        bart_tokenizer: AutoTokenizer,
        device: torch.device,
        chunk_summaries: List[str],
        model_max_tokens: int,
        target_tokens: int = 800,
    ) -> str:
        """
        Merge and polish chunk summaries with a structured prompt.
        """
        combined = "\n\n".join(chunk_summaries)
        prompt = (
            "Summarize for an analyst. Include who/what/when/where/how, numbers, and new developments. "
            "Avoid vague time words (today, recent); use absolute dates if present. No speculation.\n\n"
            "Return in this structure:\n"
            "Lede (1–2 sentences).\n"
            "• Key facts — actors, dates, magnitudes.\n"
            "• What changed vs. prior.\n"
            "• Uncertainties / conflicting reports.\n"
            "Outlook: one line.\n\n"
            f"Source notes:\n{combined}"
        )

        # total number of tokens in the prompt
        total_tokens = len(bart_tokenizer.tokenize(combined))
        # if total_tokens is greater than model_max_tokens, we need to truncate the prompt
        if total_tokens > model_max_tokens:
            prompt = prompt[:model_max_tokens]

        gen_kwargs = {
            "max_length": min(1024, target_tokens),
            "min_length": 300,
            "num_beams": 5,
            "length_penalty": 1.0,
            "eos_token_id": bart_tokenizer.eos_token_id,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
        }
        return NLPUtils.summarize_text(
            bart_model, bart_tokenizer, device, prompt, model_max_tokens, gen_kwargs
        )

    @staticmethod
    def run_quality_gates(
        summary: str, must_keep: List[str], source_entities: Dict[str, List[str]]
    ) -> Dict[str, object]:
        checks = {
            "entity_coverage": True,
            "temporal": True,
            "numeric": True,
            "redundancy_ok": True,
            "missing_entities": [],
        }
        low_sum = summary.lower()
        # entity coverage: at least half of must_keep present
        present = [e for e in must_keep if e in low_sum]
        if len(present) < max(1, len(must_keep) // 2):
            checks["entity_coverage"] = False
            checks["missing_entities"] = [e for e in must_keep if e not in low_sum][:10]
        # temporal anchoring
        if not re.search(
            r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
            summary,
            re.I,
        ):
            if source_entities.get("DATE"):
                # inject a minimal date line
                summary += f"\nTimeline: {source_entities['DATE'][0]} (source)."
            checks["temporal"] = False
        # numeric sanity
        if re.search(r"\b\d", summary) is None and source_entities.get("CARDINAL"):
            summary += f"\nNote: Source mentioned figures such as {', '.join(source_entities['CARDINAL'][:3])}."
            checks["numeric"] = False
        # redundancy check via unique bigram ratio
        tokens = re.findall(r"\w+", summary.lower())
        bigrams = list(zip(tokens, tokens[1:]))
        uniq = len(set(bigrams))
        ratio = (uniq / max(1, len(bigrams))) if bigrams else 1.0
        checks["redundancy_ok"] = ratio >= 0.6
        checks["summary"] = summary
        return checks

    # ---------- NEW: light NER with fallbacks (People/Org/Loc/Date/Numbers) ----------
    @staticmethod
    def extract_entities(text: str, lang: str = "en") -> Dict[str, List[str]]:
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "LOC": [],
            "DATE": [],
            "CARDINAL": [],
        }
        added = False
        # Try spaCy first if available
        try:
            nlp = NLPUtils.nlp_en if lang == "en" else NLPUtils.nlp_all
            doc = nlp(text)
            for ent in getattr(doc, "ents", []):
                label = ent.label_
                if label in entities:
                    entities[label].append(ent.text)
            added = True
        except Exception:
            pass
        # Transformers pipeline fallback
        if not added:
            try:
                from transformers import pipeline

                ner = pipeline("ner", grouped_entities=True)
                for item in ner(text):  # cap for speed
                    label = item.get("entity_group", "")
                    val = item.get("word", "").strip()
                    if label and val:
                        key = (
                            "ORG"
                            if "ORG" in label
                            else (
                                "PERSON"
                                if "PER" in label
                                else ("GPE" if "LOC" in label else "")
                            )
                        )
                        if key:
                            entities[key].append(val)
                added = True
            except Exception:
                pass
        # Regex fallbacks: Dates & numbers & capitalized chunks
        date_pat = re.compile(
            r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}|\d{4}-\d{2}-\d{2})\b",
            re.I,
        )
        num_pat = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
        cap_seq = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
        entities["DATE"] += date_pat.findall(text)
        entities["CARDINAL"] += num_pat.findall(text)
        # crude cap sequences as ORG/PER/GPE candidates
        for m in cap_seq.findall(text):
            if m not in entities["PERSON"]:
                entities["PERSON"].append(m)
        # Deduplicate & limit
        for k in entities:
            seen, dedup = set(), []
            for v in entities[k]:
                vv = v.strip()
                if vv and vv.lower() not in seen:
                    seen.add(vv.lower())
                    dedup.append(vv)
            entities[k] = dedup[:50]
        return entities

    # ---------- NEW: must-keep entity set ----------
    @staticmethod
    def build_must_keep_entities(
        entities: Dict[str, List[str]], top_n: int = 15
    ) -> List[str]:
        flat = []
        for k in ("PERSON", "ORG", "GPE", "LOC"):
            flat.extend(entities.get(k, []))
        # bias dates lightly
        flat.extend(entities.get("DATE", [])[:5])
        counts = Counter([f.lower() for f in flat])
        return [w for w, _ in counts.most_common(top_n)]

    # ---------- NEW: redundancy score using TF-IDF sentence sims ----------
    @staticmethod
    def compute_redundancy(sentences: List[str]) -> float:
        if len(sentences) < 3:
            return 0.0

        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
        X = vec.fit_transform(sentences)
        sims = cosine_similarity(X)
        # average of upper triangle (excluding diag)
        n = sims.shape[0]
        s = sims.sum() - n  # remove diagonal ones
        denom = n * (n - 1)
        return max(0.0, float(s / denom))

    # ---------- NEW: adaptive compression for news with MMR and forced keep ----------
    @staticmethod
    def compress_news_adaptive(
        bart_tokenizer,
        text: str,
        model_max_tokens: int,
        prompt_prefix: str = "summarize: ",
        keep_bounds: Tuple[float, float] = (0.2, 0.4),
        must_keep: Optional[List[str]] = None,
        target_tokens: Optional[int] = None,
        lang: str = "en",
    ) -> str:
        """
        Compress news text adaptively using TF-IDF + MMR.
        - keep fraction 20–40% chosen by redundancy & length
        - force-keep sentences containing must_keep entities or fresh dates
        - enforce token budget using BART tokenizer
        - if target_tokens provided, we cap around it; otherwise cap to ~2 * model_max_tokens
        """
        must_keep = set([m.lower() for m in (must_keep or [])])

        sentences = NLPUtils.safe_sent_tokenize(text, lang)
        if not sentences:
            return text
        # pick keep_fraction
        redundancy = NLPUtils.compute_redundancy(sentences)
        token_len = len(bart_tokenizer.tokenize(text))
        lo, hi = keep_bounds
        # heuristic: more redundancy → keep less; very short → keep more
        keep_fraction = lo + (hi - lo) * (
            0.3 + 0.7 * (0.4 - min(0.4, redundancy))
        )  # map
        if token_len < int(model_max_tokens * 1.5):
            keep_fraction = min(hi, max(hi, 0.4))  # near full if already short
        keep_fraction = float(min(max(keep_fraction, lo), hi))

        # TF-IDF over sentences
        vec = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, lowercase=True, max_df=0.9
        )
        # X = vec.fit_transform(sentences)
        # base = NLPUtils.np_array(X.max(axis=1).A1)

        # ensure a stable sparse format
        X = vec.fit_transform(sentences).tocsr()

        # row-wise max → dense 1-D float array
        base = NLPUtils.np_array(X.max(axis=1).toarray()).ravel()

        wc = [max(1, len(s.split())) for s in sentences]
        length_norm = [1.0 / math.sqrt(w) for w in wc]
        has_num = [1.0 if re.search(r"\d", s) else 0.0 for s in sentences]
        pos = list(
            reversed(
                [
                    0.85 + 0.15 * (i / max(1, len(sentences) - 1))
                    for i in range(len(sentences))
                ]
            )
        )
        raw = (
            0.75 * base
            + 0.10 * NLPUtils.np_array(length_norm)
            + 0.10 * NLPUtils.np_array(has_num)
            + 0.05 * NLPUtils.np_array(pos)
        )
        # force-keep boost for must_keep entities / dates
        forced = []
        date_pat = re.compile(
            r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}|\d{4}-\d{2}-\d{2})\b",
            re.I,
        )
        for i, s in enumerate(sentences):
            ss = s.lower()
            if any(ent in ss for ent in must_keep) or date_pat.search(s):
                raw[i] += 0.25  # significant boost
                forced.append(i)
        # normalize
        raw_min, raw_max = float(raw.min()), float(raw.max())
        scores = (raw - raw_min) / (raw_max - raw_min + 1e-9)
        # select top-k by rank
        k = max(1, int(len(sentences) * keep_fraction))
        top_idx = scores.argsort()[::-1][:k].tolist()
        # ensure forced are included
        for i in forced:
            if i not in top_idx:
                top_idx.append(i)
        # MMR on top set
        X_top = X[top_idx]
        # ateet
        top_idx = NLPUtils.np_array(top_idx, dtype=int)  # ensure integer indices

        sims = cosine_similarity(X_top) if X_top.shape[0] > 1 else None
        lam = 0.7
        selected_local, remaining = [], list(range(len(top_idx)))
        while remaining:
            if not selected_local:
                selected_local.append(remaining.pop(0))
                continue
            max_s = []
            for ci in remaining:
                ms = 0.0
                for sj in selected_local:
                    ms = max(ms, sims[ci, sj])
                max_s.append(ms)

            cand_scores = scores[top_idx[remaining]]
            mmr = lam * cand_scores - (1 - lam) * NLPUtils.np_array(max_s)
            pick_pos = int(mmr.argmax())
            selected_local.append(remaining.pop(pick_pos))
        sel = [top_idx[i] for i in selected_local]
        sel.sort()
        # accumulate to token budget
        prefix_tokens = (
            len(bart_tokenizer.tokenize(prompt_prefix)) if prompt_prefix else 0
        )
        budget = int(target_tokens or (model_max_tokens * 2)) - prefix_tokens
        budget = max(256, budget)
        out, used = [], 0
        sent_token_lens = [len(bart_tokenizer.tokenize(s)) for s in sentences]
        for idx in sel:
            tlen = sent_token_lens[idx]
            if used + tlen > budget:
                break
            out.append(sentences[idx])
            used += tlen
        return (prompt_prefix + " " if prompt_prefix else "") + " ".join(out)

    # ---------- NEW: chunk by token windows with overlap ----------
    @staticmethod
    def chunk_by_tokens_overlap(
        bart_tokenizer,
        text: str,
        window_tokens: int = 950,
        overlap_tokens: int = 50,
        prompt_prefix: str = "",
    ) -> List[str]:
        """
        Chunk text by token windows with overlap.
        """
        sentences = NLPUtils.safe_sent_tokenize(text)
        if not sentences:
            return [text]
        toks = [len(bart_tokenizer.tokenize(s)) for s in sentences]
        chunks, cur, cur_len = [], [], len(bart_tokenizer.tokenize(prompt_prefix))
        i = 0
        while i < len(sentences):
            s, tl = sentences[i], toks[i]
            if cur_len + tl <= window_tokens:
                cur.append(s)
                cur_len += tl
                i += 1
            else:
                if cur:
                    chunk_text = (
                        prompt_prefix + " " if prompt_prefix else ""
                    ) + " ".join(cur)
                    chunks.append(chunk_text)
                    # build overlap window from tail
                    back = []
                    back_len = 0
                    j = len(cur) - 1
                    while j >= 0 and back_len < overlap_tokens:
                        back.insert(0, cur[j])
                        back_len += len(bart_tokenizer.tokenize(cur[j]))
                        j -= 1
                    cur = back
                    cur_len = len(bart_tokenizer.tokenize(" ".join(cur))) + len(
                        bart_tokenizer.tokenize(prompt_prefix)
                    )
                else:
                    # single long sentence fallback
                    chunks.append((prompt_prefix + " " if prompt_prefix else "") + s)
                    i += 1
                    cur, cur_len = [], len(bart_tokenizer.tokenize(prompt_prefix))

        if cur:
            chunks.append(
                (prompt_prefix + " " if prompt_prefix else "") + " ".join(cur)
            )
        return chunks

    @staticmethod
    def np_array(x, dtype=float):
        try:
            return np.array(x, dtype=dtype)
        except Exception:
            # pure Python fallback
            return type(
                "_A",
                (),
                {
                    "__iter__": lambda self: iter(x),
                    "min": lambda self: min(x),
                    "max": lambda self: max(x),
                },
            )

    @staticmethod
    def split_text_smart(text: str, max_chars: int = 2000, max_chunks: int = 0) -> list:
        """
        Split the text into chunks, as the google translator has a limit of 2000 characters?
        """
        logger.info(
            f"translator.py:[Translation] Splitting text into chunks: {max_chars} characters per chunk"
        )
        sentence_endings = re.split(r"(?<=[\.\!\?।؟。！？])\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentence_endings:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    if max_chunks > 0 and len(chunks) >= max_chunks:
                        return chunks

                if len(sentence) > max_chars:
                    for i in range(0, len(sentence), max_chars):
                        chunks.append(sentence[i : i + max_chars])
                        if max_chunks > 0 and len(chunks) >= max_chunks:
                            return chunks
                    current_chunk = ""
                else:
                    current_chunk = sentence + " "
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            if max_chunks > 0 and len(chunks) >= max_chunks:
                return chunks
        return chunks

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean the text to remove any special characters.
        """
        try:
            return unicodedata.normalize("NFKC", text).strip()
        except Exception:
            return text
