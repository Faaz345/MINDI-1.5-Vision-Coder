"""
MINDI 1.5 Vision-Coder — Health Check Route

Simple health/readiness endpoint for monitoring.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return server health status."""
    return {
        "status": "healthy",
        "model": "MINDI-1.5-Vision-Coder",
        "version": "1.5.0",
    }
