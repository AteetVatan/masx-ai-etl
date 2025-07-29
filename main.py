import os
import uvicorn
from app_api import create_app
from config.settings import get_settings

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app" if settings.api_reload else app,
        host=settings.api_host or "0.0.0.0",
        port=port,
        reload=settings.api_reload or False,
        log_level=settings.log_level.lower() if settings.log_level else "info",
    )
