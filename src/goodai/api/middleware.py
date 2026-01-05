"""
API Middleware

Enterprise middleware for:
- Correlation ID propagation
- Request/response logging
- Rate limiting
- Security headers

Good AI Philosophy: Non-invasive by default.
"""

import time
import uuid
from typing import Callable

from goodai.monitoring import get_logger, CorrelationContext

# Check if Starlette is available
try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response
    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False
    BaseHTTPMiddleware = object
    Request = None
    Response = None

logger = get_logger(__name__)


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for correlation ID propagation.

    Extracts correlation ID from X-Correlation-ID header or generates new one.
    Adds correlation ID to response headers and logging context.
    """

    HEADER_NAME = "X-Correlation-ID"
    TENANT_HEADER = "X-Tenant-ID"
    USER_HEADER = "X-User-ID"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract or generate correlation ID
        correlation_id = request.headers.get(self.HEADER_NAME) or str(uuid.uuid4())
        tenant_id = request.headers.get(self.TENANT_HEADER)
        user_id = request.headers.get(self.USER_HEADER)

        # Process request with correlation context
        with CorrelationContext(
            correlation_id=correlation_id,
            tenant_id=tenant_id,
            user_id=user_id
        ):
            response = await call_next(request)

        # Add correlation ID to response
        response.headers[self.HEADER_NAME] = correlation_id

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging.

    Logs request details, response status, and timing information.
    """

    # Paths to skip logging (health checks, etc.)
    SKIP_PATHS = {"/health", "/health/live", "/health/ready", "/health/startup"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip logging for certain paths
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        start_time = time.time()

        # Log request
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else None,
            client_ip=request.client.host if request.client else None
        )

        # Process request
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2)
            )

            # Add timing header
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            logger.error(
                "Request failed",
                exc_info=e,
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                error_type=type(e).__name__
            )
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for security headers.

    Adds standard security headers to all responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.

    For production, use Redis-based rate limiting.
    """

    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._request_counts: dict = {}
        self._window_start: dict = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = request.client.host if request.client else "unknown"

        # Check rate limit
        current_time = time.time()
        window_start = self._window_start.get(client_id, current_time)

        # Reset window if expired
        if current_time - window_start >= 60:
            self._request_counts[client_id] = 0
            self._window_start[client_id] = current_time
            window_start = current_time

        # Increment count
        count = self._request_counts.get(client_id, 0) + 1
        self._request_counts[client_id] = count

        # Check limit
        if count > self.requests_per_minute:
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": int(60 - (current_time - window_start))
                },
                headers={"Retry-After": str(int(60 - (current_time - window_start)))}
            )

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - count)
        )
        response.headers["X-RateLimit-Reset"] = str(int(window_start + 60))

        return response
