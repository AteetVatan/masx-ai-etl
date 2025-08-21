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
    trigger = str(payload.get("trigger", "manual"))

    try:
        # Warm requests used by CI/Actions to pre-pull image/models
        if payload.get("mode") == "warm" or payload.get("trigger") == "warm":
            logger.info(f"warm request received")
            return {
                "ok": True,
                "status": "warmed",
                "elapsed_sec": round(time.time() - start, 3),
            }
        
        # Worker mode requests - process assigned flashpoints
        if payload.get("mode") == "worker":
            logger.info(f"worker request received for worker {payload.get('worker_id')}")
            return await handle_worker_request(payload, start)

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
        await run_etl_pipeline(date=date, cleanup=cleanup)
        
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

async def handle_worker_request(payload: Dict[str, Any], start: float) -> Dict[str, Any]:
    """
    Handle worker mode requests - process assigned flashpoints.
    """
    try:
        worker_id = payload.get("worker_id", "unknown")
        flashpoints_data = payload.get("flashpoints", [])
        date = payload.get("date")
        cleanup = payload.get("cleanup", True)
        
        logger.info(f"Worker {worker_id} processing {len(flashpoints_data)} flashpoints")
        
        # Import ETL pipeline for worker processing
        from app.etl import ETLPipeline
        from app.etl_data.etl_models import FlashpointModel
        
        # Convert flashpoint data back to models
        flashpoints = []
        for fp_data in flashpoints_data:
            try:
                flashpoint = FlashpointModel(**fp_data)
                flashpoints.append(flashpoint)
            except Exception as e:
                logger.error(f"Error creating flashpoint model: {e}")
                continue
        
        if not flashpoints:
            return {
                "ok": False,
                "error": "No valid flashpoints to process",
                "worker_id": worker_id,
                "duration_sec": round(time.time() - start, 3),
            }
        
        # Process flashpoints using ETL pipeline
        etl_pipeline = ETLPipeline(date)
        results = []
        
        for flashpoint in flashpoints:
            try:
                result = await etl_pipeline.run_etl_pipeline(flashpoint)
                results.append(result)
                logger.info(f"Worker {worker_id} completed flashpoint {flashpoint.id}")
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing flashpoint {flashpoint.id}: {e}")
                results.append(None)
        
        logger.info(f"Worker {worker_id} completed all {len(flashpoints)} flashpoints")
        
        return {
            "ok": True,
            "worker_id": worker_id,
            "flashpoints_processed": len(flashpoints),
            "results": results,
            "duration_sec": round(time.time() - start, 3),
        }
        
    except Exception as e:
        logger.error(f"Error in worker request handler: {e}")
        return {
            "ok": False,
            "error": str(e),
            "worker_id": payload.get("worker_id", "unknown"),
            "duration_sec": round(time.time() - start, 3),
        }


if __name__ == "__main__":
    # RunPod supports async handlers - this will work correctly
    runpod.serverless.start({"handler": handler})
