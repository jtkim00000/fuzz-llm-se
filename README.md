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

## Apple Silicon smoke test

The NVIDIA GPU worker cannot be exercised on an Apple GPU. For a local API and
UI smoke test, run Ollama on the Mac with a locally available model, then set
the following in an untracked `.env` file:

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

## Repository layout

- `vllm/` — vendored vLLM source at the revision previously referenced by the
  repository's submodule pointer.
- `docker-compose.yml` — local Open WebUI service.
- `.env.example` — optional Open WebUI connection defaults.
- `profiling/` — VAST-backed benchmark preparation, AIPerf workflows, and
  deterministic retrieval/JSON accuracy evaluation. See [the setup guide](profiling/README.md).
- `docs/` — step-by-step profiling guides and the GB10 runbook.
