"""
Production middleware:
- X-Request-ID correlation ID injected on every request/response
- Hardened security headers (OWASP recommended)
- Optional bearer-token API key guard on write endpoints
"""
from __future__ import annotations

import logging
import uuid

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from askdocs.config import settings

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Stamp every request with a unique X-Request-ID for log correlation."""

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add OWASP-recommended security headers to every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)
        response.headers.update(
            {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Cache-Control": "no-store",
            }
        )
        return response


# ── API Key guard ─────────────────────────────────────────────────────────────

def require_api_key(request: Request) -> None:
    """
    FastAPI dependency that enforces bearer-token authentication when
    API_KEY is configured in settings.

    If API_KEY is empty (default), auth is disabled — suitable for local dev
    and private deployments. Always set API_KEY in production.
    """
    expected = settings.api_key
    if not expected:
        return  # auth disabled
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected: Bearer <api_key>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    provided = auth.removeprefix("Bearer ").strip()
    if provided != expected:
        logger.warning(
            "Invalid API key attempt from %s [request_id=%s]",
            request.client.host if request.client else "unknown",
            getattr(request.state, "request_id", "—"),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
