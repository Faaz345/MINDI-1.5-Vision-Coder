"""
MINDI 1.5 Vision-Coder — FastAPI Application

Main entry point for the MINDI API server.
Serves code generation, vision critique, and agent orchestration endpoints.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.generate import router as generate_router
from api.routes.health import router as health_router

app = FastAPI(
    title="MINDI 1.5 Vision-Coder API",
    description="Multimodal agentic AI code generator by MINDIGENOUS.AI",
    version="1.5.0",
)

# CORS — allow the frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(generate_router, prefix="/api", tags=["Generation"])


@app.on_event("startup")
async def startup_event() -> None:
    """Load models and initialize agents on server start."""
    # Models and agents will be initialized here in later phases
    print("[MINDI API] Server starting up...")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on server shutdown."""
    print("[MINDI API] Server shutting down...")
