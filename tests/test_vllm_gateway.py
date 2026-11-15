import asyncio
import json
import unittest

from fastapi.testclient import TestClient
import httpx

from precision_split.vllm_gateway import create_vllm_gateway


def configuration():
    return {"model": "shared-model", "max_model_len": 128, "workers": {
        name: {"url": "http://" + name, "timing": {"base_seconds": 1,
               "input_seconds_per_token": 0, "output_seconds_per_token": 0}}
        for name in ("existing", "fp8")}}


class VLLMGatewayTests(unittest.TestCase):
    def test_real_messages_tokenization_forwarding_and_release(self):
        captured = []

        def backend(request):
            captured.append(request)
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/metrics":
                return httpx.Response(200, text="")
            if request.url.path == "/tokenize":
                self.assertEqual(json.loads(request.content)["messages"], [{"role": "user", "content": "hello"}])
                return httpx.Response(200, json={"count": 7})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "shared-model"}]})
            payload = json.loads(request.content)
            self.assertNotIn("session_id", payload)
            self.assertEqual(payload["max_tokens"], 8)
            return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

        app = create_vllm_gateway(configuration(), transport=httpx.MockTransport(backend))
        with TestClient(app) as client:
            self.assertEqual(client.get("/v1/models").json()["data"][0]["id"], "shared-model")
            response = client.post("/v1/chat/completions", headers={"X-Session-ID": "session-a"}, json={
                "model": "shared-model", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 8})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-precisionsplit-worker"], "existing")
            self.assertEqual(response.headers["x-session-id"], "session-a")
            state = client.get("/state").json()
            self.assertEqual(state["sessions"]["session-a"], "existing")
            self.assertTrue(all(w["assigned_requests"] == 0 for w in state["workers"].values()))
        self.assertTrue(any(r.url.path == "/v1/chat/completions" for r in captured))

    def test_unhealthy_worker_is_excluded_and_context_overflow_is_rejected(self):
        def backend(request):
            if request.url.path == "/health":
                return httpx.Response(503 if request.url.host == "existing" else 200)
            if request.url.path == "/metrics":
                return httpx.Response(200, text="")
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 120})
            return httpx.Response(200, json={"choices": []})

        app = create_vllm_gateway(configuration(), transport=httpx.MockTransport(backend))
        with TestClient(app) as client:
            payload = {"model": "shared-model", "messages": [{"role": "user", "content": "hello"}]}
            self.assertEqual(client.post("/v1/chat/completions", json={**payload, "max_tokens": 16}).status_code, 400)
            response = client.post("/v1/chat/completions", json={**payload, "max_tokens": 4})
            self.assertEqual(response.headers["x-precisionsplit-worker"], "fp8")

    def test_worker_error_status_and_cleanup(self):
        def backend(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/metrics":
                return httpx.Response(200, text="")
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 3})
            return httpx.Response(500, json={"error": "engine failed"})

        app = create_vllm_gateway(configuration(), transport=httpx.MockTransport(backend))
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={"model": "shared-model", "messages": [{}], "max_tokens": 4})
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json(), {"error": "engine failed"})
            self.assertTrue(all(w["assigned_requests"] == 0 for w in client.get("/state").json()["workers"].values()))


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_reservation_lasts_until_stream_finishes(self):
        observations = []
        stream_closed = []
        app = None

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                observations.append((await app.state.router.snapshot())["workers"]["existing"]["assigned_requests"])
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                observations.append((await app.state.router.snapshot())["workers"]["existing"]["assigned_requests"])
                yield b'data: [DONE]\n\n'

            async def aclose(self):
                stream_closed.append(True)

        def backend(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/metrics":
                return httpx.Response(200, text="")
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 3})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

        app = create_vllm_gateway(configuration(), transport=httpx.MockTransport(backend))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
                response = await client.post("/v1/chat/completions", json={"model": "shared-model", "messages": [{}], "max_tokens": 4, "stream": True})
                self.assertTrue(response.content.endswith(b"data: [DONE]\n\n"))
                self.assertEqual(observations, [1, 1])
                self.assertTrue(stream_closed)
                self.assertTrue(all(w["assigned_requests"] == 0 for w in (await app.state.router.snapshot())["workers"].values()))

    async def test_stream_cancellation_closes_upstream_and_releases(self):
        started = asyncio.Event()
        closed = []

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                yield b"data: first\n\n"
                await asyncio.Event().wait()

            async def aclose(self):
                closed.append(True)

        def backend(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/metrics":
                return httpx.Response(200, text="")
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 3})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

        app = create_vllm_gateway(configuration(), transport=httpx.MockTransport(backend))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
                task = asyncio.create_task(client.post("/v1/chat/completions", json={"model": "shared-model", "messages": [{}], "max_tokens": 4, "stream": True}))
                await asyncio.wait_for(started.wait(), 1)
                self.assertEqual((await app.state.router.snapshot())["workers"]["existing"]["assigned_requests"], 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(closed)
                self.assertEqual((await app.state.router.snapshot())["workers"]["existing"]["assigned_requests"], 0)


if __name__ == "__main__":
    unittest.main()
