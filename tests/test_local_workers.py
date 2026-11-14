import json
import unittest

from fastapi.testclient import TestClient
import httpx

from local_workers.app import WorkerConfig, create_app


class EventStream(httpx.AsyncByteStream):
    closed = False

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        yield b'data: [DONE]\n\n'

    async def aclose(self):
        self.closed = True


class WorkerTests(unittest.TestCase):
    def test_precision_mismatch_requires_explicit_simulation(self):
        with self.assertRaisesRegex(ValueError, "explicit simulated=True"):
            WorkerConfig("fp8", "fp8")

    def test_forwarding_preserves_body_query_and_session_header(self):
        captured = []

        async def backend(request):
            captured.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "4"}}]})

        app = create_app(WorkerConfig("bf16", "bfloat16"), transport=httpx.MockTransport(backend))
        body = {"model": "llama-3.1-8b-local", "messages": [{"role": "user", "content": "2+2?"}]}
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions?trace=abc", json=body, headers={"X-Session-ID": "session-17", "X-Worker-ID": "spoofed"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(captured[0].content), body)
        self.assertEqual(captured[0].url.query, b"trace=abc")
        self.assertEqual(captured[0].headers["x-session-id"], "session-17")
        self.assertNotIn("x-worker-id", captured[0].headers)
        self.assertEqual(response.headers["x-worker-id"], "bf16")
        self.assertEqual(response.headers["x-kv-cache-actual"], "bfloat16")
        self.assertEqual(response.headers["x-precision-simulated"], "false")

    def test_fp8_facade_reports_actual_precision_in_metadata_and_headers(self):
        async def backend(request):
            return httpx.Response(200, json={"data": []})

        app = create_app(WorkerConfig("fp8", "fp8", simulated=True), transport=httpx.MockTransport(backend))
        with TestClient(app) as client:
            metadata = client.get("/worker-info").json()
            response = client.get("/v1/models")
        self.assertTrue(metadata["precision_simulated"])
        self.assertTrue(metadata["shared_backend"])
        self.assertFalse(metadata["independent_kv_cache"])
        self.assertFalse(metadata["performance_comparison_valid"])
        self.assertEqual(response.headers["x-kv-cache-requested"], "fp8")
        self.assertEqual(response.headers["x-kv-cache-actual"], "bfloat16")
        self.assertEqual(response.headers["x-precision-simulated"], "true")

    def test_stream_is_forwarded_unchanged_and_upstream_is_closed(self):
        stream = EventStream()

        async def backend(request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

        app = create_app(WorkerConfig("bf16", "bfloat16"), transport=httpx.MockTransport(backend))
        with TestClient(app) as client:
            with client.stream("POST", "/v1/chat/completions", json={"stream": True}) as response:
                content = b"".join(response.iter_bytes())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream")
        self.assertIn(b'"content":"Hello"', content)
        self.assertTrue(content.endswith(b"data: [DONE]\n\n"))
        self.assertTrue(stream.closed)

    def test_upstream_error_status_and_body_are_preserved(self):
        async def backend(request):
            return httpx.Response(404, json={"error": {"message": "Unknown model"}})

        app = create_app(WorkerConfig("fp8", "fp8", simulated=True), transport=httpx.MockTransport(backend))
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={"model": "missing"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"error": {"message": "Unknown model"}})
        self.assertEqual(response.headers["x-precision-simulated"], "true")

    def test_unavailable_backend_is_not_reported_as_healthy(self):
        async def backend(request):
            raise httpx.ConnectError("Backend is stopped", request=request)

        app = create_app(WorkerConfig("bf16", "bfloat16"), transport=httpx.MockTransport(backend))
        with TestClient(app) as client:
            health = client.get("/health")
            inference = client.post("/v1/chat/completions", json={})
        self.assertEqual(health.status_code, 503)
        self.assertFalse(health.json()["ready"])
        self.assertEqual(inference.status_code, 502)
        self.assertEqual(inference.json()["error"]["code"], 502)


if __name__ == "__main__":
    unittest.main()
