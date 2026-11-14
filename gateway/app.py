"""Forward chat completions to session-affine workers; no learned routing yet."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

LOG = logging.getLogger("gateway")
HOP_HEADERS = {"connection", "content-length", "host", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}


@dataclass(frozen=True)
class Worker:
    name: str
    url: str

    def __post_init__(self):
        parsed = urlsplit(self.url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("Worker URL must be an HTTP(S) origin without a path")


def create_app(workers: list[Worker], *, transport=None):
    if not workers or len({worker.name for worker in workers}) != len(workers):
        raise ValueError("At least one uniquely named worker is required")
    assignments: dict[str, str] = {}
    active = {worker.name: 0 for worker in workers}
    by_name = {worker.name: worker for worker in workers}
    lock = asyncio.Lock()
    next_index = 0

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5), trust_env=False, transport=transport) as client:
            app.state.client = client
            yield

    app = FastAPI(title="PrecisionSplit Gateway", lifespan=lifespan)

    @app.get("/health")
    async def health():
        # Gateway process health, not a claim that workers are healthy.
        return {"status": "ok", "workers": len(workers)}

    @app.get("/v1/models")
    async def models(request: Request):
        token = os.environ.get("GATEWAY_WORKER_TOKEN")
        headers = {"authorization": f"Bearer {token}"} if token else {}
        try:
            response = await request.app.state.client.get(
                workers[0].url.rstrip("/") + "/v1/models", headers=headers
            )
        except httpx.HTTPError:
            return JSONResponse({"error": {"message": "Worker model list unavailable", "type": "upstream_error"}}, status_code=502)
        return Response(content=response.content, status_code=response.status_code,
                        media_type=response.headers.get("content-type", "application/json"))

    @app.post("/v1/chat/completions")
    async def completion(request: Request):
        nonlocal next_index
        session = request.headers.get("x-session-id", "").strip()
        if not session or len(session) > 256:
            return JSONResponse({"error": {"message": "X-Session-ID is required (maximum 256 characters)", "type": "invalid_request_error"}}, status_code=400)
        body = await request.body()
        # Least outstanding requests, round-robin for ties. This is a baseline,
        # not a claim about true GPU load, latency, health, or precision.
        async with lock:
            name = assignments.get(session)
            if name is None:
                min_active = min(active.values())
                for offset in range(len(workers)):
                    index = (next_index + offset) % len(workers)
                    if active[workers[index].name] == min_active:
                        name = workers[index].name
                        next_index = (index + 1) % len(workers)
                        break
                assignments[session] = name
            active[name] += 1
        worker = by_name[name]
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS | {"authorization", "x-routed-node", "x-policy"}}
        headers["accept-encoding"] = "identity"
        token = os.environ.get("GATEWAY_WORKER_TOKEN")
        if token:
            headers["authorization"] = f"Bearer {token}"
        url = worker.url.rstrip("/") + "/v1/chat/completions"
        if request.url.query:
            url += "?" + request.url.query
        LOG.info("route worker=%s policy=least_outstanding", name)
        try:
            upstream = await request.app.state.client.send(request.app.state.client.build_request("POST", url, headers=headers, content=body), stream=True)
        except httpx.HTTPError as exc:
            async with lock:
                active[name] -= 1
            status = 504 if isinstance(exc, httpx.TimeoutException) else 502
            # Do not reroute: a request might already have reached the worker,
            # and an existing session's cache cannot be transferred.
            return JSONResponse({"error": {"message": "Assigned worker unavailable; session remains pinned", "type": "upstream_error"}}, status_code=status, headers={"X-Routed-Node": name, "X-Policy": "least_outstanding"})
        response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS | {"x-routed-node", "x-policy"}}
        response_headers.update({"X-Routed-Node": name, "X-Policy": "least_outstanding"})

        async def forward():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                async with lock:
                    active[name] -= 1

        return StreamingResponse(forward(), status_code=upstream.status_code, headers=response_headers)

    return app


def app_from_env():
    local_url = os.environ.get("GATEWAY_LOCAL_OLLAMA_URL")
    if local_url:
        return create_app([Worker("local-ollama", local_url)])
    return create_app([
        Worker("bf16", os.environ.get("GATEWAY_BF16_URL", "http://127.0.0.1:8001")),
        Worker("fp8", os.environ.get("GATEWAY_FP8_URL", "http://127.0.0.1:8002")),
    ])
