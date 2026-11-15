"""Exercise the active production request handler using the real trained forest."""

import asyncio
import json
import unittest

import httpx

from precision_split.vllm_gateway import create_vllm_gateway


def config():
    return {"model": "shared-model", "max_model_len": 8192, "workers": {
        name: {"url": "http://" + name, "effective_kv_dtype": dtype,
               "timing": {"base_seconds": 9999, "input_seconds_per_token": 0,
                          "output_seconds_per_token": 0}}
        for name, dtype in (("bf16", "bfloat16"), ("fp8", "fp8"))}}


class PredictiveHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.unhealthy = set()
        self.calls = []
        owner = self

        class Stream(httpx.AsyncByteStream):
            def __init__(self, hold):
                self.hold = hold

            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
                if self.hold:
                    owner.started.set()
                    await owner.release.wait()
                yield b'data: [DONE]\n\n'

        async def backend(request):
            if request.url.path == "/health":
                return httpx.Response(503 if request.url.host in self.unhealthy else 200)
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 512})
            if request.url.path == "/metrics":
                return httpx.Response(200, text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0\n")
            body = json.loads(request.content)
            prompt = body["messages"][0]["content"]
            self.calls.append(request.url.host)
            if prompt == "fail":
                raise httpx.ConnectError("controlled failure", request=request)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream(prompt == "hold"))

        self.app = create_vllm_gateway(config(), transport=httpx.MockTransport(backend))
        self.context = self.app.router.lifespan_context(self.app)
        await self.context.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://gateway")

    async def asyncTearDown(self):
        self.release.set()
        await self.client.aclose()
        await self.context.__aexit__(None, None, None)

    async def send(self, session, prompt="quick"):
        return await self.client.post("/v1/chat/completions", headers={"X-Session-ID": session}, json={
            "model": "shared-model", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32, "stream": True, "chat_template_kwargs": {"enable_thinking": False}})

    async def counts(self):
        state = (await self.client.get("/state")).json()
        return {k: v["assigned_requests"] for k, v in state["workers"].items()}

    async def test_idle_load_affinity_failure_and_stream_cleanup(self):
        with self.assertLogs("precision_split.routing", level="INFO") as logs:
            idle = await self.send("idle")
            self.assertEqual(idle.headers["x-precisionsplit-worker"], "fp8")
            held = asyncio.create_task(self.send("held", "hold"))
            try:
                await asyncio.wait_for(self.started.wait(), 2)
                self.assertEqual(await self.counts(), {"bf16": 0, "fp8": 1})
                overflow = await self.send("overflow")
                self.assertEqual(overflow.headers["x-precisionsplit-worker"], "bf16")
                followup = await self.send("held")
                self.assertEqual(followup.headers["x-precisionsplit-worker"], "fp8")
                failed = await self.send("failed", "fail")
                self.assertEqual(failed.status_code, 502)
                self.assertEqual(await self.counts(), {"bf16": 0, "fp8": 1})
            finally:
                self.release.set()
                response = await held
            self.assertIn("[DONE]", response.text)
            self.assertEqual(await self.counts(), {"bf16": 0, "fp8": 0})
        decisions = [json.loads(line.split(":", 2)[-1]) for line in logs.output]
        decisions = {d["session_id"]: d for d in decisions if d["event"] == "assignment"}
        for session in ("idle", "overflow"):
            d = decisions[session]
            expected = min(d["scores"], key=lambda w: (d["scores"][w]["predicted_completion_ms"], w))
            self.assertEqual(d["worker"], expected)
            self.assertEqual(d["reason"], "minimum_predicted_completion_time")
        d = decisions["overflow"]["scores"]["fp8"]
        self.assertEqual(d["features"]["client_outstanding_requests"], 1)
        self.assertEqual(d["features"]["running_requests"], 0)  # Lagging telemetry must not hide reservations.
        self.assertEqual(decisions["held"]["reason"], "session_affinity")

    async def test_cancel_and_unhealthy_pinned_worker(self):
        held = asyncio.create_task(self.send("pinned", "hold"))
        await asyncio.wait_for(self.started.wait(), 2)
        self.unhealthy.add("fp8")
        r = await self.send("pinned")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(await self.counts(), {"bf16": 0, "fp8": 1})
        held.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await held
        self.assertEqual(await self.counts(), {"bf16": 0, "fp8": 0})

    async def test_invalid_prediction_falls_back_and_releases(self):
        self.app.state.predictor.artifact["models"]["e2e_ms"] = {"constant": float("nan")}
        with self.assertLogs("precision_split.routing", level="INFO") as logs:
            response = await self.send("fallback")
        self.assertEqual(response.status_code, 200)
        self.assertIn('"reason": "least_outstanding_requests"', "\n".join(logs.output))
        self.assertEqual(await self.counts(), {"bf16": 0, "fp8": 0})

    async def test_worker_specific_context_limit_excludes_faster_candidate(self):
        self.app.state.predictor.workers["fp8"]["manifest"] = {"max_model_len": 512}
        response = await self.send("fits-only-bf16")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-precisionsplit-worker"], "bf16")

    async def test_concurrent_reservations_see_prior_load_and_ties_are_deterministic(self):
        router = self.app.state.router
        seen = []

        def scores(counts):
            seen.append(dict(counts))
            return {w: {"predicted_completion_ms": 100 + n * 100} for w, n in counts.items()}

        assignments = await asyncio.gather(*(router.reserve(str(i), 512, 32, score_candidates=scores) for i in range(6)))
        self.assertEqual([a.worker for a in assignments], ["bf16", "fp8"] * 3)
        self.assertEqual(seen[1], {"bf16": 1, "fp8": 0})
        for a in assignments:
            await router.release(a)
        self.assertEqual(await self.counts(), {"bf16": 0, "fp8": 0})
