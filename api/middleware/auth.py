"""
MINDI 1.5 Vision-Coder — Auth Middleware

API key validation for production deployment.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> str:
    """Validate the API key from request headers."""
    expected_key = os.environ.get("MINDI_API_KEY", "")

    # In development, skip auth if no key is configured
    if not expected_key:
        return "dev-mode"

    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key
