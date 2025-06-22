from etl import ETLPipeline
from singleton import ChromaClientSingleton

"""
HDBSCAN for smart clustering
ChromaDB for efficient vector ops
BART/T5 for text summarization
Modular, singleton-backed architecture
All aligned with real-world scale and performance in MASX AI
"""


if __name__ == "__main__":
    # centralize the cleanup right before invoking all of them
    print("Deleting all tracked Chroma collections before pipeline runs...")
    ChromaClientSingleton.cleanup_chroma()
    ETLPipeline.run_etl_pipeline()