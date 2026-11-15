# PrecisionSplit on two GB10 nodes: Qwen3.8 27B

NVIDIA's [Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
runs on two GPU workers behind a one-core CPU gateway. For the local mock
demo, see [precision_split/README.md](../../precision_split/README.md).

Verified November 14, 2026: four-request routing smoke tests passed from
the Mac and inside the cluster, and all 21 local tests passed. Both concurrent
warm requests completed in about eight seconds; this is not a comparative
benchmark. The initial cold cluster test hit its 120-second client timeout
while BF16 kernels compiled. Reservations cleared and both warm reruns passed.

| Service | Workflow ID | Node | Resources |
| --- | --- | --- | --- |
| BF16 attention KV | `c8b096fd-97a8-44f2-91bd-7d60db53c916` | CIQ-CHI-SPARK03 | 1 GPU, 8 CPU cores, 80 GiB requested memory |
| FP8 attention KV | `95e2dc41-c219-4cde-8fb3-64858ca262e4` | CIQ-CHI-SPARK04 | 1 GPU, 8 CPU cores, 80 GiB requested memory |
| Gateway | `eb835a92-fe1d-4f44-b402-471f1bdd5e48` | CIQ-CHI-SPARK04 | 1 CPU core, 2 GiB, no GPU |

## Same checkpoint, different KV caches

Both API model IDs are `nvidia/Qwen3.8-27B-NVFP4`. Both engines use vLLM
0.28.0 and revision `482ca0f3832238542f8f5295dde86b5f22711d80`.
The existing `qwen38-27b-shared` persistent volume is mounted at `/models`;
the checkpoint directory is
`/models/nvidia--Qwen3.8-27B-NVFP4/482ca0f3832238542f8f5295dde86b5f22711d80`.
Its existing `VERIFIED.json` records 21,945,291,730 bytes, 19 files and
completion/checksums. No new download was needed for this replacement.

Weights use NVIDIA's mixed NVFP4/FP8 quantization. Both workers have
`--dtype bfloat16`; this does not convert quantized weights to full BF16.
Their attention KV settings are `--kv-cache-dtype bfloat16` and
`--kv-cache-dtype fp8`. The FP8 runtime resolves storage to
`torch.float8_e4m3fn`. KV dtype does not set every recurrent-state or
kernel-arithmetic dtype in this hybrid model.

Common settings: 8,192-token context; maximum 4 sequences and 2,048 batched
tokens per scheduler iteration; 8 GiB cache allocation; tensor/pipeline
parallelism 1; eager execution; text-only mode; prefix caching and chunked
prefill enabled; no speculative decoding. Explicit cache bytes take precedence
over GPU memory utilization. BF16 selects FLASH_ATTN; FP8 selects FLASHINFER,
a backend difference to record in later comparisons. FP8 KV scale quality,
accuracy and precision speedup have not been established by smoke testing.

## Connect and test from the Mac

API base URL:

```text
https://endpoint-8bf62dd6e7fb917c58362ab0b19d729ee1e4598b.endpoints.fb-hack.civilfritz.net/v1
```

This is a Fuzzball user-scoped endpoint. Refresh its two-hour token using the
existing Fuzzball login, from the repo root:

```sh
.venv-precisionsplit/bin/python deploy/fuzzball/authorize_endpoint.py \
  --endpoint-id endpoint-8bf62dd6e7fb917c58362ab0b19d729ee1e4598b \
  --output .local-inference/precision-split/qwen38-gateway-token
```

The token file has mode 0600. The approved cluster CA is at
`.local-inference/precision-split/cluster-ca.crt`. Both files are ignored by
Git; TLS verification remains enabled. API clients send an Authorization
bearer token. Use the team's approved channel for credentials on another
machine; credential values do not belong in chat, source, or a manifest.

```sh
.venv-precisionsplit/bin/python scripts/test_precision_split_gpu.py \
  --url https://endpoint-8bf62dd6e7fb917c58362ab0b19d729ee1e4598b.endpoints.fb-hack.civilfritz.net \
  --token-file .local-inference/precision-split/qwen38-gateway-token \
  --ca-file .local-inference/precision-split/cluster-ca.crt \
  --disable-thinking \
  --output results/precision-split-qwen38/mac-smoke.json
```

The test sends two concurrent 64-token-capped requests, an 8-token follow-up
and an 8-token streaming request. It checks both workers, session affinity,
a completed stream and zero remaining reservations. Thinking is disabled
for these short tests through `chat_template_kwargs.enable_thinking=false`.
Otherwise it is a request-level choice; a tiny output cap can be consumed by
reasoning before an answer appears. This test is not a performance study.

## Routing on the greedy-gateway branch

New sessions select the healthy worker with fewer outstanding requests,
including running and queued requests. Ties rotate between workers. Counts
are reserved atomically before HTTP forwarding and released on completion,
failure or cancellation; streaming holds its count until closure. Follow-ups
stay on the pinned worker and count toward its load too. No timing estimator
or regression output influences this decision. Existing timing fields are
retained for diagnostics and mock delays.

This balances counts, not execution cost. One long request still counts as
one, and session affinity can make the split uneven. Worker-direct traffic
is not counted. Keep one gateway process while load/session state is local.

The live deployment recorded above was verified with the previous timing-based
policy. This branch change takes effect only after rendering and redeploying
the gateway; the existing cluster service is not changed by editing the repo.

## API, configuration and deployment

- `GET /health`: both backend health checks; 200 when at least one is healthy.
- `GET /state`: reservations, session pins, health, model and estimate source.
- `GET /v1/models`: exact served model discovery.
- `POST /v1/chat/completions`: messages, tools and streaming.

Supply `X-Session-ID` for affinity or reuse its returned value. Responses
also include `X-PrecisionSplit-Worker` (`bf16` or `fp8`) and
`X-PrecisionSplit-Request-ID`.

[precision-split-qwen38.json](precision-split-qwen38.json) configures
`http://192.168.99.12:18027` (BF16) and `http://192.168.99.13:18027` (FP8).
The gateway listens on 19038. These private node addresses must be checked
and updated after rescheduling. From the Mac use the authenticated HTTPS URL.

[qwen38-bf16-kv.fz](qwen38-bf16-kv.fz) and
[qwen38-fp8-kv.fz](qwen38-fp8-kv.fz) reproduce the running workers.
Each requests one GPU. Placement put them on separate nodes; these files
do not contain unique node-affinity labels. Do not start duplicate services
while current allocations are running.

```sh
.venv-precisionsplit/bin/python deploy/fuzzball/render_precision_gateway.py \
  --output results/precision-split-qwen38/gateway-next.fz
fuzzball workflow validate results/precision-split-qwen38/gateway-next.fz
```

Rendering/validation does not update a running service. Replacement needs a
new deployment, port planning and new endpoint credentials. Source/config
are embedded in the Fuzzfile. The CPU gateway reuses the cached vLLM image's
HTTP dependencies without loading a model or requesting a GPU.

See [routing.py](../../precision_split/routing.py) for atomic reservations,
[forwarding.py](../../precision_split/forwarding.py) for HTTP forwarding and
[vllm_gateway.py](../../precision_split/vllm_gateway.py) for the real API.
Local records are in `results/precision-split-qwen38/`; object-cache copies
use `fb://group/precision-split/eb835a92-fe1d-4f44-b402-471f1bdd5e48/`
with seven-day TTL. Credentials are excluded.

Superseded Qwen2.5 workflows: `34067792-1bbd-42fe-9c94-11373c9fcef4`
(FP16/auto KV), `6778fb91-dcca-4ca9-b5d0-143a0356cf39` (FP8 KV), and
gateway `56378649-c8fa-4de9-8982-b66dc1fb7fce`.
All three superseded workflows are confirmed canceled. Historical files
`qwen-fp8-kv.fz` and `precision-split-qwen.json` still describe Qwen2.5;
current renderer defaults use Qwen3.8.
