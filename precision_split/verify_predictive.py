"""Verify the production gateway over real HTTP using controlled worker stubs."""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import socket

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from .vllm_gateway import create_vllm_gateway


async def verify():
    gate, started = asyncio.Event(), asyncio.Event()
    events = []

    class Decisions(logging.Handler):
        def emit(self, record):
            events.append(json.loads(record.getMessage()))

    def worker():
        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"ready": True}

        @app.get("/metrics")
        async def metrics():
            # Deliberately lagging telemetry: gateway reservations must still matter.
            return Response("vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0\n")

        @app.post("/tokenize")
        async def tokenize():
            return {"count": 512}

        @app.get("/v1/models")
        async def models():
            return {"data": [{"id": "controlled-model"}]}

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            payload = await request.json()
            prompt = payload["messages"][0]["content"]
            if prompt == "fail":
                return JSONResponse({"error": "controlled upstream failure"}, status_code=500)

            async def chunks():
                yield b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
                if prompt == "hold":
                    started.set()
                    await gate.wait()
                yield b'data: [DONE]\n\n'

            return StreamingResponse(chunks(), media_type="text/event-stream")

        return app

    sockets, servers, tasks = [], [], []
    logger = logging.getLogger("precision_split.routing")
    handler, old_level = Decisions(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    held = None
    try:
        for _ in range(3):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        urls = [f"http://127.0.0.1:{sock.getsockname()[1]}" for sock in sockets]
        config = {"model": "controlled-model", "max_model_len": 8192, "workers": {
            name: {"url": url, "effective_kv_dtype": dtype,
                   "timing": {"base_seconds": 9999, "input_seconds_per_token": 0,
                              "output_seconds_per_token": 0}}
            for name, dtype, url in zip(("bf16", "fp8"), ("bfloat16", "fp8"), urls)}}
        gateway = create_vllm_gateway(config)
        for app, sock in zip((worker(), worker(), gateway), sockets):
            server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
            servers.append(server)
            tasks.append(asyncio.create_task(server.serve(sockets=[sock])))
        for _ in range(200):
            if all(s.started for s in servers):
                break
            if any(t.done() for t in tasks):
                raise RuntimeError("HTTP service exited before readiness")
            await asyncio.sleep(.01)
        else:
            raise RuntimeError("HTTP service readiness timed out")

        async with httpx.AsyncClient(base_url=urls[2], timeout=15, trust_env=False) as client:
            async def send(session, prompt="quick"):
                return await client.post("/v1/chat/completions", headers={"X-Session-ID": session}, json={
                    "model": "controlled-model", "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 32, "stream": True, "chat_template_kwargs": {"enable_thinking": False}})

            async def state():
                r = await client.get("/state")
                r.raise_for_status()
                return r.json()

            idle = await send("idle")
            idle.raise_for_status()
            assert idle.headers["x-precisionsplit-worker"] == "fp8"
            held = asyncio.create_task(send("held", "hold"))
            await asyncio.wait_for(started.wait(), 5)
            busy_state = await state()
            assert busy_state["workers"]["fp8"]["assigned_requests"] == 1
            overflow = await send("overflow")
            overflow.raise_for_status()
            assert overflow.headers["x-precisionsplit-worker"] == "bf16"
            followup = await send("held")
            followup.raise_for_status()
            assert followup.headers["x-precisionsplit-worker"] == "fp8"
            failed = await send("failed", "fail")
            assert failed.status_code == 500
            after_failure = await state()
            assert {w: s["assigned_requests"] for w, s in after_failure["workers"].items()} == {"bf16": 0, "fp8": 1}
            gate.set()
            stream = await held
            stream.raise_for_status()
            assert "[DONE]" in stream.text
            final = await state()
            assert all(s["assigned_requests"] == 0 for s in final["workers"].values())
            decisions = [e for e in events if e["event"] == "assignment"]
            for d in decisions:
                if d["reason"] == "minimum_predicted_completion_time":
                    expected = min(d["scores"], key=lambda w: (d["scores"][w]["predicted_completion_ms"], w))
                    assert d["worker"] == expected
            return {"passed": True, "gateway_url": urls[2], "endpoint": "/v1/chat/completions",
                    "model_version": gateway.state.predictor.version, "decisions": decisions,
                    "failure_status": failed.status_code, "final_state": final,
                    "scope": "Real HTTP sockets and production handler, controlled workers, actual saved forest; not a GPU benchmark"}
    finally:
        gate.set()
        if held is not None and not held.done():
            held.cancel()
            await asyncio.gather(held, return_exceptions=True)
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)
        for sock in sockets:
            sock.close()
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(verify())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    for d in report["decisions"]:
        print(d["session_id"], {w: s.get("predicted_completion_ms") for w, s in d["scores"].items()},
              "->", d["worker"], d["reason"])
    print("PASS: actual model, HTTP routing, load shift, affinity, upstream failure, stream lifetime, and zero reservations")


if __name__ == "__main__":
    main()
