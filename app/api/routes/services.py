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


import re
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.config.logging_config import get_api_logger
from main_etl import run_etl_pipeline

router = APIRouter()
logger = get_api_logger("ServiceRoutes")


@router.post("/run")
async def run_etl(background_tasks: BackgroundTasks, date: Optional[str] = None):
    """
    Get status of all services.

    Returns:
        Dictionary with service status information
    """
    logger.info("services.py:Services status requested")

    try:
        logger.info(f"services.py:Running ETL pipeline for date: {date}")

        if date:
            # ETL logic for a specific date
            logger.info(f"services.py:Running ETL for date: {date}")
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
                raise HTTPException(
                    status_code=400, detail="Date must be in format YYYY-MM-DD"
                )

            background_tasks.add_task(run_etl_pipeline, date)
        else:
            # ETL logic for default (e.g., today)
            logger.info("services.py:Running ETL with default date")
            background_tasks.add_task(run_etl_pipeline)

        # if date is not in format ("%Y-%m-%d") raise error

        return {"status": "ETL pipeline started in background"}
    except Exception as e:
        logger.error(f"services.py:Services status retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Services status retrieval failed: {str(e)}"
        )
