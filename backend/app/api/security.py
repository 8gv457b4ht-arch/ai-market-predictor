"""API-key authentication, per-client rate limiting and security headers (pure ASGI middleware)."""
from __future__ import annotations

import hmac
import json
import threading
import time

PUBLIC_PATHS = ("/api/health", "/api/ready", "/api/config")


def _client_ip(scope, trust_proxy: bool) -> str:
    if trust_proxy:
        for k, v in scope.get("headers", []):
            if k == b"x-forwarded-for":
                return v.decode().split(",")[0].strip()
            if k == b"x-real-ip":
                return v.decode().strip()
    client = scope.get("client")
    return client[0] if client else "unknown"


async def _send_json(send, status: int, body: dict, extra_headers: list | None = None) -> None:
    data = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers + (extra_headers or [])})
    await send({"type": "http.response.body", "body": data})


class TokenBucket:
    def __init__(self, per_minute: int):
        self.capacity = max(1, per_minute)
        self.rate = self.capacity / 60.0
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1:
                self.buckets[key] = (tokens - 1, now)
                return True, 0.0
            self.buckets[key] = (tokens, now)
            if len(self.buckets) > 10_000:  # bound memory
                self.buckets = {k: v for k, v in self.buckets.items() if now - v[1] < 120}
            return False, (1 - tokens) / self.rate


class SecurityMiddleware:
    def __init__(self, app, api_key: str, per_minute: int, trust_proxy: bool):
        self.app = app
        self.api_key = api_key
        self.bucket = TokenBucket(per_minute)
        self.trust_proxy = trust_proxy

    def _authorized(self, scope) -> bool:
        if not self.api_key:
            return True
        headers = dict(scope.get("headers", []))
        token = headers.get(b"x-api-key", b"").decode()
        auth = headers.get(b"authorization", b"").decode()
        if not token and auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        return bool(token) and hmac.compare_digest(token.encode(), self.api_key.encode())

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if path.startswith("/api/"):
            ok, wait = self.bucket.allow(_client_ip(scope, self.trust_proxy))
            if not ok:
                return await _send_json(send, 429, {"error": "rate_limited", "retry_after_sec": round(wait, 1)},
                                        [(b"retry-after", str(int(wait) + 1).encode())])
            if path not in PUBLIC_PATHS and not self._authorized(scope):
                return await _send_json(send, 401, {"error": "unauthorized",
                                                    "detail": "send the API key as X-API-Key or Bearer token"})

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                h = list(message.get("headers", []))
                h += [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
                      (b"referrer-policy", b"no-referrer"),
                      (b"content-security-policy",
                       b"default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
                       b"connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")]
                if path.startswith("/api/"):
                    h.append((b"cache-control", b"no-store"))
                message["headers"] = h
            await send(message)
        return await self.app(scope, receive, send_with_headers)
