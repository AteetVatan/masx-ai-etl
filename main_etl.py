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

import asyncio
from typing import Optional, List
from app.etl import ETLPipeline
from app.singleton import ChromaClientSingleton
from app.config import get_service_logger
from app.enumeration import WorkerEnums
from app.services import ProxyService

logger = get_service_logger("ETLPipeline")

"""
HDBSCAN for smart clustering
ChromaDB for efficient vector ops
BART/T5 for text summarization
Modular, singleton-backed architecture
All aligned with real-world scale and performance in MASX AI
"""


async def run_etl_pipeline(
    trigger: str = WorkerEnums.COORDINATOR.value,
    date: Optional[str] = None,
    flashpoints_ids: List[str] = None,
    cleanup: bool = True,
):
    # centralize the cleanup right before invoking all of them
    # print("Deleting all tracked Chroma collections before pipeline runs...")
    logger.info(f"main_etl.py:run_etl_pipeline called")
    if cleanup:
        ChromaClientSingleton.cleanup_chroma()
        
    if trigger == WorkerEnums.COORDINATOR.value:
        proxy_service = ProxyService()
        await proxy_service.ping_start_proxy()
        
    trigger = WorkerEnums.ETL_WORKER.value
    date = "2025-07-01"
    flashpoints_ids = ["70ef3f5a-3dbd-4b9a-8eb5-1b971a37fbc0"]    
        

    etl_pipeline = ETLPipeline(date)
    await etl_pipeline.run_all_etl_pipelines(
        trigger=trigger, flashpoints_ids=flashpoints_ids
    )


def run_etl_pipeline_sync(
    trigger: str = WorkerEnums.COORDINATOR.value,
    date: Optional[str] = None,
    flashpoints_ids: List[str] = None,
    cleanup: bool = True,
):
    """Synchronous wrapper for backward compatibility."""
    return asyncio.run(
        run_etl_pipeline(
            trigger=trigger, date=date, flashpoints_ids=flashpoints_ids, cleanup=cleanup
        )
    )


if __name__ == "__main__":
    run_etl_pipeline_sync()
