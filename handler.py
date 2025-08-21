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

import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
import runpod
from app.config import get_service_logger

logger = get_service_logger("Handler")

# Make sure root is importable
sys.path.append(os.environ.get("PYTHONPATH", "/app"))

# If it's /app/app/main_etl.py, use:
# from app.main_etl import run_etl_pipeline

# Defaults via env
ALLOW_CLEANUP = os.environ.get("MASX_ETL_CLEANUP", "true").lower() == "true"


def _parse_date(s: Optional[str]) -> str:
    """Accept 'YYYY-MM-DD' or ISO strings; fallback to today."""
    if not s:
        return datetime.now().strftime("%Y-%m-%d")
    # Basic YYYY-MM-DD validation
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        # Attempt strict YYYY-MM-DD
        try:
            return datetime.strptime(s, "%Y-%m-%d").date().isoformat()
        except Exception as e:
            raise ValueError(f"invalid 'date': {s} ({e})")


async def handler(job: Dict[str, Any]):
    """
    RunPod serverless entrypoint (Async version).
    
    RunPod supports async handlers, which is the correct approach
    for our async ETL pipeline to avoid nested event loop conflicts.

    Expected inputs:
      {
        "date": "YYYY-MM-DD",         # optional; defaults to today
        "trigger": "GSG|cron|manual", # optional, for logging
        "cleanup": true,              # optional; default from env
        "mode": "warm"                # optional; warm pull/preload
      }
    """
    start = time.time()
    payload = job.get("input") or {}
    trigger = str(payload.get("trigger", "coordinator"))
    flashpoints_ids = payload.get("flashpoints", None)

    try:
        # Warm requests used by CI/Actions to pre-pull image/models
        if payload.get("mode") == "warm" or payload.get("trigger") == "warm":
            logger.info(f"warm request received")
            return {
                "ok": True,
                "status": "warmed",
                "elapsed_sec": round(time.time() - start, 3),
            }
        
      
        date = _parse_date(payload.get("date"))
        cleanup = bool(payload.get("cleanup", ALLOW_CLEANUP))

        # lazy import to avoid boot-time crashes
        from main_etl import run_etl_pipeline

        logger.info(
            f"running async handler for ETL pipeline for date: {date} and cleanup: {cleanup}"
        )
        
        # Get worker configuration
        from app.config import get_settings
        settings = get_settings()
        num_workers = settings.runpod_workers
        
        logger.info(f"Using {num_workers} workers for parallel execution")
        
        # Directly await the async ETL pipeline - no nested event loops!
        await run_etl_pipeline(trigger=trigger, date=date, flashpoints_ids=flashpoints_ids, cleanup=cleanup)
        
        logger.info(f"async handler completed")
        return {
            "ok": True,
            "date": date,
            "trigger": trigger,
            "cleanup": cleanup,
            "workers": num_workers,
            "duration_sec": round(time.time() - start, 3),
        }

    except Exception as e:
        logger.error(f"Error in async handler: {e}")
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc()[-2000:],
            "trigger": trigger,
            "duration_sec": round(time.time() - start, 3),
        }


if __name__ == "__main__":
    # RunPod supports async handlers - this will work correctly
    runpod.serverless.start({"handler": handler})
