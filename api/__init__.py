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

"""
FastAPI API layer for Global Signal Grid (MASX) Agentic AI System.

Provides REST API endpoints for:
- System monitoring and health checks
- Workflow management and execution
- Data retrieval and analysis
- Service status and configuration

Usage:
    from app.api import create_app

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from .app import create_app
from .routes import services

__all__ = [
    "create_app",
    "services",
]
