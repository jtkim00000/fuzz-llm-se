"""Forward requests to one local model while exposing honest worker metadata."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx
from starlette.background import BackgroundTask


HOP_HEADERS = {
    "connection", "content-length", "host", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "proxy-connection", "te", "trailer",
    "transfer-encoding", "upgrade",
}
IDENTITY_HEADERS = {
    "x-worker-id", "x-test-only", "x-kv-cache-requested", "x-kv-cache-actual",
    "x-precision-simulated", "x-shared-backend",
}


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: str
    requested_kv_cache_dtype: str
    simulated: bool = False
    backend_url: str = "http://127.0.0.1:8000"
    actual_kv_cache_dtype: str = "bfloat16"

    def __post_init__(self):
        parsed = urlsplit(self.backend_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("backend_url must be an HTTP(S) server address")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("backend_url must omit /v1, query parameters, and fragments")
        if self.requested_kv_cache_dtype != self.actual_kv_cache_dtype and not self.simulated:
            raise ValueError("A precision mismatch requires explicit simulated=True")

    def headers(self):
        return {
            "X-Worker-ID": self.worker_id,
            "X-Test-Only": "true",
            "X-KV-Cache-Requested": self.requested_kv_cache_dtype,
            "X-KV-Cache-Actual": self.actual_kv_cache_dtype,
            "X-Precision-Simulated": str(self.simulated).lower(),
            "X-Shared-Backend": "true",
        }

    def metadata(self):
        return {
            "application": "precision-worker-test-facade",
            "process_id": os.getpid(),
            "worker_id": self.worker_id,
            "test_only": True,
            "requested_kv_cache_dtype": self.requested_kv_cache_dtype,
            "actual_kv_cache_dtype": self.actual_kv_cache_dtype,
            "precision_simulated": self.simulated,
            "backend_url": self.backend_url.rstrip("/"),
            "shared_backend": True,
            "independent_kv_cache": False,
            "independent_gpu_queue": False,
            "performance_comparison_valid": False,
        }


def create_app(config: WorkerConfig, *, transport=None):
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            base_url=config.backend_url.rstrip("/"),
            timeout=httpx.Timeout(120, connect=5),
            trust_env=False,
            transport=transport,
        ) as client:
            app.state.backend = client
            yield

    app = FastAPI(title=f"Local test worker: {config.worker_id}", lifespan=lifespan)

    @app.get("/worker-info")
    async def worker_info():
        return JSONResponse(config.metadata(), headers=config.headers())

    @app.get("/health")
    async def health(request: Request):
        try:
            response = await request.app.state.backend.get("/health", timeout=5)
            ready = response.status_code == 200
        except httpx.HTTPError:
            ready = False
        return JSONResponse(
            {**config.metadata(), "ready": ready},
            status_code=200 if ready else 503,
            headers=config.headers(),
        )

    async def forward(request: Request):
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in HOP_HEADERS | IDENTITY_HEADERS
        }
        headers["accept-encoding"] = "identity"
        path = request.url.path
        if request.url.query:
            path += "?" + request.url.query
        client = request.app.state.backend
        upstream_request = client.build_request(
            request.method, path, headers=headers, content=await request.body(),
        )
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            status = 504 if isinstance(exc, httpx.TimeoutException) else 502
            return JSONResponse(
                {"error": {"message": "Local model backend is unavailable", "type": "upstream_error", "code": status}},
                status_code=status,
                headers=config.headers(),
            )
        response_headers = {
            key: value for key, value in upstream.headers.items()
            if key.lower() not in HOP_HEADERS | IDENTITY_HEADERS
        }
        response_headers.update(config.headers())

        if "text/event-stream" in upstream.headers.get("content-type", ""):
            async def chunks():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()

            return StreamingResponse(
                chunks(), status_code=upstream.status_code,
                headers=response_headers, background=BackgroundTask(upstream.aclose),
            )
        try:
            content = await upstream.aread()
            return Response(content, status_code=upstream.status_code, headers=response_headers)
        finally:
            await upstream.aclose()

    for path in ("/v1/models", "/metrics", "/version"):
        app.add_api_route(path, forward, methods=["GET"])
    for path in ("/v1/chat/completions", "/v1/completions", "/tokenize", "/detokenize"):
        app.add_api_route(path, forward, methods=["POST"])
    return app
