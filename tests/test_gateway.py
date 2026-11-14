import unittest

import httpx

from gateway.app import Worker, create_app


class MockStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"token":"hi"}\n\n'
        yield b'data: [DONE]\n\n'


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []

        async def upstream(request):
            self.calls.append((request.url.host, request.headers.get("x-session-id")))
            if request.url.host == "dead":
                raise httpx.ConnectError("offline", request=request)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "test-model"}]})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=MockStream())

        self.app = create_app([Worker("bf16", "http://one"), Worker("fp8", "http://two")], transport=httpx.MockTransport(upstream))
        self.context = self.app.router.lifespan_context(self.app)
        await self.context.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://gateway")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.context.__aexit__(None, None, None)

    async def test_session_affinity_and_stream(self):
        first = await self.client.post("/v1/chat/completions", headers={"X-Session-ID": "one"}, json={"stream": True})
        second = await self.client.post("/v1/chat/completions", headers={"X-Session-ID": "two"}, json={"stream": True})
        followup = await self.client.post("/v1/chat/completions", headers={"X-Session-ID": "one"}, json={"stream": True})
        self.assertEqual([first.headers["x-routed-node"], second.headers["x-routed-node"], followup.headers["x-routed-node"]], ["bf16", "fp8", "bf16"])
        self.assertIn(b"data: [DONE]", first.content)
        self.assertEqual(self.calls, [("one", "one"), ("two", "two"), ("one", "one")])

    async def test_missing_session_is_rejected(self):
        response = await self.client.post("/v1/chat/completions", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.calls, [])

    async def test_models_are_discovered_from_worker(self):
        response = await self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "test-model")

    async def test_failed_worker_does_not_move_session(self):
        app = create_app([Worker("bf16", "http://dead"), Worker("fp8", "http://two")], transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("offline", request=request))))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
                a = await client.post("/v1/chat/completions", headers={"X-Session-ID": "same"}, json={})
                b = await client.post("/v1/chat/completions", headers={"X-Session-ID": "same"}, json={})
        self.assertEqual((a.status_code, b.status_code), (502, 502))
        self.assertEqual((a.headers["x-routed-node"], b.headers["x-routed-node"]), ("bf16", "bf16"))
