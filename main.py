import os
import uvicorn
from app.api.app import create_app
from app.config.settings import get_settings

print("Starting API...")
app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting API on port {port}...")

    # uvicorn.run(
    #     "main:app" if settings.api_reload else app,
    #     host=settings.api_host or "0.0.0.0",
    #     port=port,
    #     reload=settings.api_reload or False,
    #     log_level=settings.log_level.lower() if settings.log_level else "info",
    # )

    uvicorn_args = {
        "host": settings.api_host or "0.0.0.0",
        "port": port,
        "log_level": (settings.log_level or "info").lower(),
    }

    print(
        f"API will run on {uvicorn_args['host']}:{uvicorn_args['port']} (log level: {uvicorn_args['log_level']})"
    )

    if settings.api_reload:
        uvicorn.run("main:app", reload=True, **uvicorn_args)
    else:
        uvicorn.run(app, **uvicorn_args)
