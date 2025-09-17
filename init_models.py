import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.config import Settings  

settings = Settings()
cache = Path(settings.model_cache_dir)
cache.mkdir(parents=True, exist_ok=True)

print(f"📦 Caching models into {cache}")

# 1. Embedding model
SentenceTransformer("sentence-transformers/all-mpnet-base-v2", cache_folder=str(cache))
print("✅ all-mpnet-base-v2")

# 2. BART summarizer
AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn", cache_dir=str(cache))
AutoTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=str(cache))
print("✅ bart-large-cnn")

# 3. NLLB translator
AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", cache_dir=str(cache))
AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", cache_dir=str(cache))
print("✅ nllb-200-distilled-600M")

print("🎉 All models cached successfully in", cache)
