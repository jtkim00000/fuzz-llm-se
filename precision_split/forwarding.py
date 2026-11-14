"""HTTP transport; has no routing policy or load accounting."""

import os

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


class VLLMForwarder(WorkerForwarder):
    """Real vLLM transport, using shared-model tokenization and streaming HTTP."""

    def __init__(self, workers, *, timeout=120.0, transport=None):
        super().__init__({name: info["url"] for name, info in workers.items()},
                         timeout=timeout, transport=transport)
        self.headers = {}
        for name, info in workers.items():
            key_env = info.get("api_key_env")
            self.headers[name] = {"Authorization": "Bearer " + os.environ[key_env]} if key_env else {}

    async def healthy(self, worker):
        try:
            response = await self.client.get(self.urls[worker] + "/health",
                                             headers=self.headers[worker], timeout=5)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def model_ids(self, worker):
        response = await self.client.get(self.urls[worker] + "/v1/models", headers=self.headers[worker])
        response.raise_for_status()
        return [entry["id"] for entry in response.json()["data"]]

    async def token_count(self, worker, model, payload):
        body = {"model": model, "messages": payload["messages"], "add_generation_prompt": True}
        for key in ("tools", "chat_template", "chat_template_kwargs"):
            if key in payload:
                body[key] = payload[key]
        response = await self.client.post(self.urls[worker] + "/tokenize", json=body, headers=self.headers[worker])
        response.raise_for_status()
        count = response.json()["count"]
        if not isinstance(count, int) or count < 0:
            raise ValueError("Worker returned invalid token count")
        return count

    async def start_chat(self, worker, payload):
        request = self.client.build_request("POST", self.urls[worker] + "/v1/chat/completions",
                                            json=payload, headers=self.headers[worker])
        return await self.client.send(request, stream=True)
