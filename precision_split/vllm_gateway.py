"""OpenAI-compatible gateway for individually addressable real vLLM workers."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from .forwarding import VLLMForwarder
from .routing import GreedyRouter, WorkerUnavailable
from .timing import ServiceTime


logger = logging.getLogger("precision_split.vllm")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[dict] = Field(min_length=1)
    stream: bool = False
    session_id: str | None = Field(default=None, min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)


def create_vllm_gateway(config, *, transport=None):
    workers = config["workers"]
    model = config["model"]
    router = GreedyRouter({name: ServiceTime(**info["timing"]) for name, info in workers.items()})

    @asynccontextmanager
    async def lifespan(app):
        app.state.forwarder = VLLMForwarder(workers, timeout=config.get("worker_timeout", 120), transport=transport)
        try:
            yield
        finally:
            await app.state.forwarder.close()

    app = FastAPI(title="PrecisionSplit vLLM gateway", lifespan=lifespan)
    app.state.router = router

    async def availability():
        names = list(workers)
        values = await asyncio.gather(*(app.state.forwarder.healthy(name) for name in names))
        return dict(zip(names, values))

    @app.get("/health")
    async def health():
        healthy = await availability()
        return JSONResponse({"ready": any(healthy.values()), "workers": healthy},
                            status_code=200 if any(healthy.values()) else 503)

    @app.get("/state")
    async def state():
        return {**await router.snapshot(), "healthy": await availability(), "model": model,
                "worker_urls": {name: info["url"] for name, info in workers.items()},
                "estimate_source": config.get("estimate_source", "uncalibrated_initial_heuristic")}

    @app.get("/v1/models")
    async def models():
        healthy = await availability()
        available = False
        for name, ready in healthy.items():
            if ready:
                try:
                    available = model in await app.state.forwarder.model_ids(name)
                    if available:
                        break
                except (httpx.HTTPError, ValueError, KeyError):
                    continue
        return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "precisionsplit"}] if available else []}

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: Request):
        if body.model != model:
            return JSONResponse({"error": {"message": "Unknown model", "type": "invalid_request_error"}}, status_code=404)
        if body.max_tokens is not None and body.max_completion_tokens is not None:
            return JSONResponse({"error": {"message": "Specify one output-token limit"}}, status_code=400)
        session = request.headers.get("X-Session-ID") or body.session_id or uuid.uuid4().hex
        payload = body.model_dump(exclude_none=True, exclude={"session_id"})
        budget = body.max_completion_tokens or body.max_tokens or 128
        if body.max_tokens is None and body.max_completion_tokens is None:
            payload["max_tokens"] = budget
        healthy = await availability()
        eligible = [name for name, ready in healthy.items() if ready]
        if not eligible:
            return JSONResponse({"error": {"message": "No healthy workers"}}, status_code=503)
        try:
            tokens = await app.state.forwarder.token_count(eligible[0], model, payload)
            if tokens + budget > config.get("max_model_len", 32768):
                return JSONResponse({"error": {"message": "Input plus output budget exceeds context limit"}}, status_code=400)
            assignment = await router.reserve(session, tokens, budget, eligible_workers=eligible)
        except WorkerUnavailable as error:
            return JSONResponse({"error": {"message": str(error)}}, status_code=503)
        except (httpx.HTTPError, KeyError, ValueError):
            return JSONResponse({"error": {"message": "Worker tokenization failed"}}, status_code=502)

        started = time.monotonic()
        upstream = None
        released = False

        async def cleanup(outcome="finished"):
            nonlocal released
            if released:
                return
            released = True
            try:
                if upstream is not None:
                    await upstream.aclose()
            finally:
                await asyncio.shield(router.release(assignment, outcome))
                logger.info(json.dumps({"event": "request_end", "request_id": assignment.request_id,
                    "worker": assignment.worker, "session_id": session, "outcome": outcome,
                    "input_tokens": tokens, "output_budget": budget,
                    "seconds": time.monotonic() - started}))

        headers = {"X-PrecisionSplit-Worker": assignment.worker, "X-Session-ID": session,
                   "X-PrecisionSplit-Request-ID": assignment.request_id}
        try:
            upstream = await app.state.forwarder.start_chat(assignment.worker, payload)
            headers["Content-Type"] = upstream.headers.get("content-type", "application/json")
            if not body.stream or upstream.status_code >= 400:
                try:
                    content = await upstream.aread()
                    return Response(content, status_code=upstream.status_code, headers=headers)
                finally:
                    await cleanup("finished" if upstream.status_code < 400 else "upstream_error")

            async def chunks():
                outcome = "stream_closed_or_cancelled"
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                    outcome = "finished"
                finally:
                    await cleanup(outcome)

            headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            return StreamingResponse(chunks(), status_code=upstream.status_code, headers=headers,
                                     background=BackgroundTask(cleanup))
        except BaseException as error:
            await cleanup("forwarding_failed")
            if isinstance(error, httpx.HTTPError):
                return JSONResponse({"error": {"message": "Worker forwarding failed"}},
                                    status_code=504 if isinstance(error, httpx.TimeoutException) else 502,
                                    headers={k: v for k, v in headers.items() if k != "Content-Type"})
            raise

    return app
