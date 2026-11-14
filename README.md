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

## Repository layout

- `vllm/` — vendored vLLM source at the revision previously referenced by the
  repository's submodule pointer.
- `docker-compose.yml` — local Open WebUI service.
- `.env.example` — optional Open WebUI connection defaults.
