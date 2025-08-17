# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# │  Project: MASX AI – Strategic Agentic AI System              │
# │  All rights reserved.                                        │
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


from app.etl import ETLPipeline
from app.singleton import ChromaClientSingleton
from typing import Optional
from app.config import get_service_logger

logger = get_service_logger("ETLPipeline")

"""
HDBSCAN for smart clustering
ChromaDB for efficient vector ops
BART/T5 for text summarization
Modular, singleton-backed architecture
All aligned with real-world scale and performance in MASX AI
"""


def run_etl_pipeline(date: Optional[str] = None, cleanup: bool = True):
    # centralize the cleanup right before invoking all of them
    # print("Deleting all tracked Chroma collections before pipeline runs...")
    logger.info(f"run_etl_pipeline called")
    if cleanup:
        ChromaClientSingleton.cleanup_chroma()

    etl_pipeline = ETLPipeline(date)
    etl_pipeline.run_all_etl_pipelines()


if __name__ == "__main__":
    run_etl_pipeline()
