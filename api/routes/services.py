# # ┌───────────────────────────────────────────────────────────────┐
# # │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# # │  Project: MASX AI – Strategic Agentic AI System              │
# # │  All rights reserved.                                        │
# # └───────────────────────────────────────────────────────────────┘
# #
# # MASX AI is a proprietary software system developed and owned by Ateet Vatan Bahmani.
# # The source code, documentation, workflows, designs, and naming (including "MASX AI")
# # are protected by applicable copyright and trademark laws.
# #
# # Redistribution, modification, commercial use, or publication of any portion of this
# # project without explicit written consent is strictly prohibited.
# #
# # This project is not open-source and is intended solely for internal, research,
# # or demonstration use by the author.
# #
# # Contact: ab@masxai.com | MASXAI.com


from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from config.logging_config import get_api_logger
from main_etl import run_etl_pipeline

router = APIRouter()
logger = get_api_logger("ServiceRoutes")


@router.post("/run")
async def run_etl(background_tasks: BackgroundTasks):
    """
    Get status of all services.

    Returns:
        Dictionary with service status information
    """
    logger.info("Services status requested")

    try:
        background_tasks.add_task(run_etl_pipeline)
        return {"status": "ETL pipeline started in background"}
    except Exception as e:
        logger.error(f"Services status retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Services status retrieval failed: {str(e)}"
        )
