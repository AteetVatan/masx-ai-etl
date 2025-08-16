# handler.py
import os, sys, time, traceback
import runpod
from datetime import datetime

# Make sure "app" package can be imported even if PYTHONPATH isn't set
sys.path.append(os.environ.get("PYTHONPATH", "/app"))

from .main_etl import run_etl_pipeline  # uses your ETLPipeline internally

DEFAULT_TIMEOUT_SEC = int(os.environ.get("MASX_ETL_TIMEOUT_SEC", "3300"))  # ~55m
ALLOW_CLEANUP = os.environ.get("MASX_ETL_CLEANUP", "true").lower() == "true"

def run(job):
    """
    input: {
        "date": "YYYY-MM-DD",       # optional; if None, ETL decides (e.g., today)
        "trigger": "GSG|cron|manual",  # optional, for logging
        "cleanup": true              # optional; override default cleanup behavior
      }
    """
    start = time.time()
    payload = job.get("input") or {}
    date = payload.get("date")
    trigger = payload.get("trigger", "manual")
    cleanup = payload.get("cleanup", ALLOW_CLEANUP)

    try:
                
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Execute your ETL
        run_etl_pipeline(date=date, cleanup=cleanup)
        return {
            "ok": True, "date": date, "trigger": trigger, "cleanup": cleanup,
            "duration_sec": round(time.time() - start, 3)
        }
    except Exception as e:
        return {
            "ok": False, "error": str(e),
            "trace": traceback.format_exc()[-2000:],
            "date": date, "trigger": trigger,
            "duration_sec": round(time.time() - start, 3)
        }

runpod.serverless.start({"handler": run})
