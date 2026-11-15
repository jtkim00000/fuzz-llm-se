"""HTTP applications for the shared FIFO mock worker and gateway."""

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import json
import logging
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, Field

from .forwarding import WorkerForwarder
from .routing import GreedyRouter


logger = logging.getLogger("precision_split.worker")


class InferenceRequest(BaseModel):
    session_id: str | None = Field(default=None, min_length=1)
    input_length: int = Field(ge=0, strict=True)
    expected_output_length: int = Field(ge=0, strict=True)


class WorkerRequest(InferenceRequest):
    request_id: str = Field(min_length=1)


@dataclass
class Job:
    request: WorkerRequest
    sequence: int
    enqueued_at: float
    future: asyncio.Future


class MockWorker:
    def __init__(self, mode, timing, *, sleep=asyncio.sleep):
        self.mode = mode
        self.timing = timing
        self.sleep = sleep
        self.queue = asyncio.Queue()
        self.running = None
        self.accepted = 0
        self.completed = 0

    async def run(self):
        while True:
            job = await self.queue.get()
            self.running = job
            started = time.monotonic()
            duration = self.timing.estimate(job.request.input_length, job.request.expected_output_length)
            logger.info(json.dumps({"event": "worker_start", "worker": self.mode,
                                    "request_id": job.request.request_id, "sequence": job.sequence}))
            try:
                await self.sleep(duration)
                result = {
                    "mock": True, "worker": self.mode, "request_id": job.request.request_id,
                    "session_id": job.request.session_id, "sequence": job.sequence,
                    "response": f"Dummy response from {self.mode}",
                    "service_seconds": duration,
                    "queue_wait_seconds": started - job.enqueued_at,
                    "processing_seconds": time.monotonic() - started,
                }
                if not job.future.done():
                    job.future.set_result(result)
                self.completed += 1
            except asyncio.CancelledError:
                job.future.cancel()
                raise
            finally:
                self.running = None
                self.queue.task_done()

    async def enqueue(self, request):
        self.accepted += 1
        future = asyncio.get_running_loop().create_future()
        self.queue.put_nowait(Job(request, self.accepted, time.monotonic(), future))
        return await asyncio.shield(future)

    def snapshot(self):
        return {"mock": True, "mode": self.mode, "timing": self.timing.as_dict(),
                "running_request_id": self.running.request.request_id if self.running else None,
                "queued_requests": self.queue.qsize(), "accepted": self.accepted,
                "completed": self.completed}


def create_worker(mode, timings, *, sleep=asyncio.sleep):
    worker = MockWorker(mode, timings[mode], sleep=sleep)

    @asynccontextmanager
    async def lifespan(app):
        runner = asyncio.create_task(worker.run())
        try:
            yield
        finally:
            runner.cancel()
            with suppress(asyncio.CancelledError):
                await runner
            while not worker.queue.empty():
                job = worker.queue.get_nowait()
                job.future.cancel()
                worker.queue.task_done()

    app = FastAPI(title=f"PrecisionSplit mock {mode}", lifespan=lifespan)
    app.state.worker = worker

    @app.get("/health")
    @app.get("/state")
    async def state():
        return worker.snapshot()

    @app.post("/infer")
    async def infer(request: WorkerRequest):
        return await worker.enqueue(request)

    return app


def create_gateway(timings, urls, *, timeout=30.0, transport=None):
    router = GreedyRouter(timings)

    @asynccontextmanager
    async def lifespan(app):
        forwarder = WorkerForwarder(urls, timeout=timeout, transport=transport)
        app.state.forwarder = forwarder
        try:
            yield
        finally:
            await forwarder.close()

    app = FastAPI(title="PrecisionSplit gateway", lifespan=lifespan)
    app.state.router = router

    @app.get("/health")
    async def health():
        return {"ready": True, "mock": True, "worker_health_checked": False}

    @app.get("/state")
    async def state():
        return {**await router.snapshot(), "worker_urls": urls}

    @app.post("/infer")
    async def infer(request: InferenceRequest):
        session_id = request.session_id or uuid.uuid4().hex
        assignment = await router.reserve(session_id, request.input_length, request.expected_output_length)
        payload = {**request.model_dump(), "session_id": session_id, "request_id": assignment.request_id}
        outcome = "failed_or_cancelled"
        try:
            result = await app.state.forwarder.infer(assignment.worker, payload)
            outcome = "finished"
            return {"routing": assignment.as_dict(), "result": result}
        except (httpx.HTTPError, ValueError) as error:
            status = 504 if isinstance(error, httpx.TimeoutException) else 502
            return JSONResponse({"error": "worker_call_failed", "error_type": type(error).__name__,
                                 "routing": assignment.as_dict()}, status_code=status)
        finally:
            await asyncio.shield(router.release(assignment, outcome))

    return app
