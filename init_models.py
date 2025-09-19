import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.config import Settings  

settings = Settings()
cache = Path(settings.model_cache_dir)
cache.mkdir(parents=True, exist_ok=True)

print(f"📦 Caching models into {cache}")

# local_files_only=False during preload, which ensures 
# all required model + tokenizer files 
# (including tokenizer_config.json, special_tokens_map.json, etc.) 
# are pulled from Hugging Face Hub into your cache.

# 1. Embedding model
SentenceTransformer("sentence-transformers/all-mpnet-base-v2", 
                    cache_folder=str(cache),
                    local_files_only=False)
print("all-mpnet-base-v2")

# 2. BART summarizer
AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn", 
                                      cache_dir=str(cache), 
                                      local_files_only=False)
AutoTokenizer.from_pretrained("facebook/bart-large-cnn", 
                              cache_dir=str(cache), 
                              local_files_only=False)
print("bart-large-cnn")

# 3. NLLB translator
AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", 
                                      cache_dir=str(cache), 
                                      local_files_only=False)
AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", 
                              cache_dir=str(cache), 
                              local_files_only=False)
print("nllb-200-distilled-600M")

print("All models cached successfully in", cache)
