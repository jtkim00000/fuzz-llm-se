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

## Alternative Ollama smoke test

The NVIDIA GPU worker cannot be exercised on an Apple GPU. For an alternative
local API and UI smoke test, run Ollama on the Mac with a locally available
model, then set the following in an untracked `.env` file:

```dotenv
OPENAI_API_BASE_URL=http://host.docker.internal:11434/v1
OPENAI_API_KEY=ollama
```

Start the UI with `docker compose up -d open-webui`. For example, use
`ollama run llama3.1:8b` to check local inference. This tests the UI and API
connection, not this repository's vLLM engine, CUDA kernels, or GB10
performance. A persisted Open WebUI connection may need to be updated in its
admin settings. Keep `.env`, model weights, and benchmark outputs out of Git.

For the cluster, pin the model, vLLM revision, image, KV-cache dtype and
hardware settings before profiling. Verify BF16 and FP8 support on the actual
GB10 workers and expose their endpoints to the gateway; Fuzzball workflow
definitions and cluster access still need to be supplied.

For the BF16-versus-FP8 KV-cache comparison, use the
[stock-engine launch profiles](docs/precision-profiles.md). The FP8 worker uses
`configs/kv-cache/fp8.yaml`; the matched BF16 worker uses
`configs/kv-cache/bf16.yaml`. Both profiles keep the model compute dtype at
BF16 and use the same checkpoint and engine build.

## Repository layout

- `vllm/` — vendored vLLM source at the revision previously referenced by the
  repository's submodule pointer.
- `docker-compose.yml` — local Open WebUI service.
- `.env.example` — optional Open WebUI connection defaults.
- `local_workers/` — localhost test facades with explicit precision metadata.
- `scripts/local_workers.py` — start, inspect, or stop the test facades.
- `requirements-local-workers.txt` — pinned dependencies for the test facades.
- `profiling/` — AIPerf smoke test and notes on collecting router data.
- `docs/` — step-by-step profiling guides and the GB10 runbook.
