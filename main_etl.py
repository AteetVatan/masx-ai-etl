from etl import ETLPipeline
from singleton import ChromaClientSingleton
from typing import Optional

"""
HDBSCAN for smart clustering
ChromaDB for efficient vector ops
BART/T5 for text summarization
Modular, singleton-backed architecture
All aligned with real-world scale and performance in MASX AI
"""


def run_etl_pipeline(date: Optional[str] = None):
    # centralize the cleanup right before invoking all of them
    print("Deleting all tracked Chroma collections before pipeline runs...")
    ChromaClientSingleton.cleanup_chroma()
    etl_pipeline = ETLPipeline(date)
    etl_pipeline.run_all_etl_pipelines()


# if __name__ == "__main__":
#     run_etl_pipeline()
