"""HTTP transport; has no routing policy or load accounting."""

import httpx


class WorkerForwarder:
    def __init__(self, urls, *, timeout=30.0, transport=None):
        self.urls = {}
        for worker, url in urls.items():
            parsed = httpx.URL(url)
            if parsed.scheme not in ("http", "https") or not parsed.host:
                raise ValueError("Worker URLs must be HTTP(S) addresses")
            if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
                raise ValueError("Worker URLs must be base URLs without a path or query")
            self.urls[worker] = str(parsed).rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False, transport=transport)

    async def infer(self, worker, payload):
        response = await self.client.post(self.urls[worker] + "/infer", json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()
