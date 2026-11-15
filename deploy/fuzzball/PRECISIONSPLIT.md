# Running PrecisionSplit on the two GB10 nodes

Deployment verified November 14, 2026. This is real Qwen inference through
PrecisionSplit, with two GPU workers and a CPU gateway. The local mock demo
remains available separately in [`precision_split/README.md`](../../precision_split/README.md).

| Service | Fuzzball workflow ID | Node | Resources |
| --- | --- | --- | --- |
| Existing Qwen worker | `34067792-1bbd-42fe-9c94-11373c9fcef4` | CIQ-CHI-SPARK04 | 1 GPU, 8 CPU cores, 34 GiB requested memory |
| New FP8-KV Qwen worker | `6778fb91-dcca-4ca9-b5d0-143a0356cf39` | CIQ-CHI-SPARK03 | 1 GPU, 8 CPU cores, 34 GiB requested memory |
| PrecisionSplit gateway | `56378649-c8fa-4de9-8982-b66dc1fb7fce` | CIQ-CHI-SPARK04 | 1 CPU core, 2 GiB memory, no GPU allocation |

Both engines serve `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` using stock vLLM 0.28.0.
They mount the existing persistent `qwen25-awq` volume at `/data` and use
`/data/qwen--qwen2.5-coder-7b-instruct-awq`. No second model download was started.
The original worker was not restarted or reconfigured.

The original worker reports FP16 compute, automatic KV dtype and FLASH_ATTN.
The new worker reports FP16 compute, FP8 KV, and FLASHINFER; its runtime log
resolves KV storage to `torch.float8_e4m3fn` on `sm121`. This is **not** a matched
BF16-versus-FP8 experiment. FP8 calibration and comparative correctness remain
unverified. It also uses the existing Qwen checkpoint rather than the proposed
Nemotron NVFP4 checkpoint.

## Connect from the Mac

The gateway API base URL is:

```text
https://endpoint-fb03aca68aaefd98969be3de947e134c26f2b448.endpoints.fb-hack.civilfritz.net/v1
```

The endpoint has Fuzzball `user` scope. A short-lived endpoint token was saved
on this Mac at `.local-inference/precision-split/gateway-token`, with mode
`0600`. The approved cluster root CA is at
`.local-inference/precision-split/cluster-ca.crt`; it was retrieved through
authenticated workflow execution from Fuzzball's published node trust store.
These files are ignored by Git. No certificate verification was disabled and
the Mac's global trust configuration was not changed.

To refresh the token using the existing Fuzzball login, from the repo root:

```sh
.venv-precisionsplit/bin/python deploy/fuzzball/authorize_endpoint.py \
  --endpoint-id endpoint-fb03aca68aaefd98969be3de947e134c26f2b448 \
  --output .local-inference/precision-split/gateway-token
```

The default lifetime is two hours. The helper prints only the protected file
path and expiry duration. API clients present the token in the Authorization
bearer header. Obtain the approved CA and access token through the team's
approved channel when connecting from another machine; neither belongs in a
manifest or source file.

Run the same four-request smoke test used for verification:

```sh
.venv-precisionsplit/bin/python scripts/test_precision_split_gpu.py \
  --url https://endpoint-fb03aca68aaefd98969be3de947e134c26f2b448.endpoints.fb-hack.civilfritz.net \
  --token-file .local-inference/precision-split/gateway-token \
  --ca-file .local-inference/precision-split/cluster-ca.crt \
  --output results/precision-split-cluster/mac-smoke.json
```

This sends two concurrent 64-token-capped requests, one 8-token-capped follow-up
and one 8-token-capped streaming request. It checks that both workers handle
traffic, affinity holds, streaming ends with `[DONE]`, and reservations return
to zero. This is a functional smoke test, not a performance or accuracy study.

## API and routing

- `GET /health`: live checks of both backends; HTTP 200 when at least one is healthy.
- `GET /state`: reservation counts, session assignments, health, model and estimate source.
- `GET /v1/models`: served model discovery.
- `POST /v1/chat/completions`: normal chat payloads, including tools and streaming.

Use the exact model ID above. Supply `X-Session-ID` on requests for conversation
affinity, or reuse the generated ID returned in the response's `X-Session-ID`.
The response also carries `X-PrecisionSplit-Worker` (`existing` or `fp8`) and
`X-PrecisionSplit-Request-ID`. The gateway does not store conversation messages;
clients must send the required history.

The real adapter checks backend health, tokenizes the actual messages, reserves
load atomically, and forwards to the selected worker. Streaming bytes pass
through unchanged; the reservation stays active until the stream closes.
An unhealthy pinned worker produces 503 rather than silently migrating the
session. The gateway does not automatically retry generation.

Both workers currently have the same initial timing heuristic:

```text
service seconds = 0.05 + 0.001 * input tokens + 0.04 * output budget
```

The coefficients are **uncalibrated**. The gateway uses the existing serial
remaining-work approximation; real vLLM continuously batches requests. Routing
mechanics are verified, but prediction quality, saturation behavior, optimality
and any precision speedup have not been established. Replace the coefficients
and load model with benchmark-derived estimates and telemetry before those
claims. Keep one gateway process while its reservation/session state is local.

## Configuration and deployment source

[`precision-split-qwen.json`](precision-split-qwen.json) contains the worker
addresses and estimates. The verified private-cluster routes are
`http://192.168.99.13:18000` and `http://192.168.99.12:18000`. These direct routes
avoid an intermediary load balancer. After worker rescheduling, verify placement
and update the routes as needed; these are node addresses, not service discovery.

[`qwen-fp8-kv.fz`](qwen-fp8-kv.fz) reproduces the new worker. Do not submit a
duplicate while its GPU allocation is already running.

[`render_precision_gateway.py`](render_precision_gateway.py) packages the local
Python sources and configuration into a Fuzzball workflow:

```sh
.venv-precisionsplit/bin/python deploy/fuzzball/render_precision_gateway.py \
  --output results/precision-split-cluster/gateway-next.fz
fuzzball workflow validate results/precision-split-cluster/gateway-next.fz
```

Generating/validating a file does not update the running service. Coordinate a
gateway replacement before submitting a new copy, including port placement.
The current gateway reuses the cached vLLM container's HTTP dependencies but
does not load a model or request a GPU. Source/configuration are embedded in
the submitted workflow, so subsequent local edits require a new deployment.

The real adapter lives in [`vllm_gateway.py`](../../precision_split/vllm_gateway.py)
and `VLLMForwarder` in [`forwarding.py`](../../precision_split/forwarding.py).
Shared atomic reservation/affinity logic remains in
[`routing.py`](../../precision_split/routing.py).

The deployment record, cluster/Mac smoke results, startup evidence and routing
logs are under `results/precision-split-cluster/` on this Mac. No credentials are
included in those artifacts.

Deployment and cluster smoke records are also saved in the Fuzzball object
cache under `fb://group/precision-split/56378649-c8fa-4de9-8982-b66dc1fb7fce/`
with a seven-day TTL. Use `object_get` or `object_download` through the MCP to
retrieve them from another session.
