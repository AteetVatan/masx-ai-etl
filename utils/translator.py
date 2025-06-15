"""
This module provides utilities to normalize and translate multilingual text to English.
"""

import re
import time
import unicodedata
from deep_translator import GoogleTranslator


class Translator:
    """
    Provides utilities to normalize and translate multilingual text to English.
    """

    def __init__(self, max_chars=2000, retries=3, delay=2):
        self.max_chars = max_chars
        self.retries = retries
        self.delay = delay

    def translate_to_english(self, text: str) -> str:
        """
        Translate the text to English.
        """
        try:
            # split the text into chunks, as the google translator has a limit of 2000 characters?
            chunks = self.__split_text_smart(text)
            translated_chunks = [
                self.__safe_translate(self.__clean_text(chunk)) for chunk in chunks
            ]
            return "\n\n".join(translated_chunks)
        except Exception as e:
            print("[CRITICAL] Full translation failed:", e)
            return text

    def __split_text_smart(self, text: str) -> list:
        """
        Split the text into chunks, as the google translator has a limit of 2000 characters?
        """
        sentence_endings = re.split(r"(?<=[\.\!\?।؟。！？])\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentence_endings:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) + 1 <= self.max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(sentence) > self.max_chars:
                    for i in range(0, len(sentence), self.max_chars):
                        chunks.append(sentence[i : i + self.max_chars])
                    current_chunk = ""
                else:
                    current_chunk = sentence + " "
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def __clean_text(self, chunk: str) -> str:
        return unicodedata.normalize("NFKC", chunk).strip()

    def __safe_translate(self, chunk: str) -> str:
        for attempt in range(self.retries):
            try:
                return GoogleTranslator(source="auto", target="en").translate(chunk)
            except Exception as e:
                print(f"[Attempt {attempt + 1}/{self.retries}] Translation failed: {e}")
                time.sleep(self.delay)
        print(f"[Fallback] Using original chunk:\n{chunk[:80]}...")
        return chunk
