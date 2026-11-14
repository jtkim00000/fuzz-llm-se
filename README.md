# Fuzzball Quantization LLM Serving Engine

This repository is the starting point for a precision-aware gateway in front of
multiple vLLM workers. The vLLM source is vendored at [`vllm/`](vllm/), so the
project can be extended and tested against a pinned engine revision without a
submodule checkout.

## Open WebUI

Open WebUI is included as a Docker Compose service. Start it with:

```bash
docker compose up -d open-webui
```

Then open <http://localhost:3000>. In **Settings → Admin → Connections**, add
the gateway or vLLM OpenAI-compatible endpoint, ending in `/v1`.

The included Compose file uses a persistent `open-webui` volume and exposes
the service on port 3000. The vLLM worker services are intentionally not
started by default: they require a model, GPU allocation, and cluster-specific
Fuzzball configuration that will be added with the router.

## Local worker tests

Two local HTTP facades are available for router testing: BF16 at
`http://127.0.0.1:8001/v1` and an explicitly simulated FP8 facade at
`http://127.0.0.1:8002/v1`. Both use the existing Llama server on port 8000.
The installed Metal backend does not support FP8 KV caches; both facades use
the same actual BF16 cache and do not provide independent precision benchmarks.
See [local worker instructions](docs/local-workers.md) for startup, inspection,
and tests.

## Repository layout

- `vllm/` — vendored vLLM source at the revision previously referenced by the
  repository's submodule pointer.
- `docker-compose.yml` — local Open WebUI service.
- `.env.example` — optional Open WebUI connection defaults.
- `local_workers/` — localhost test facades with explicit precision metadata.
- `scripts/local_workers.py` — start, inspect, or stop the test facades.
- `requirements-local-workers.txt` — pinned dependencies for the test facades.
