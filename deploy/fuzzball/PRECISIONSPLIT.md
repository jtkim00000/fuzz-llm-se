# PrecisionSplit on two GB10 nodes: Qwen3.8 27B

NVIDIA's [Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
runs on two GPU workers behind a one-core CPU gateway. For the local mock
demo, see [precision_split/README.md](../../precision_split/README.md).

Updated November 15, 2026: the BF16 and FP8 workers were canceled. Two
explicit FP16 attempts were subsequently canceled by another user/session,
which launched the stock replacements below. At the user's request, these
stock replacements are left running. Runtime logs report BF16 compute and
automatic KV dtype. The original gateway workflow and endpoint are preserved.
Startup status is `started`; this replacement pair has not been smoke-tested
in this session. Earlier routing results apply to the previous worker pair.

| Service | Workflow ID | Node | Resources |
| --- | --- | --- | --- |
| Stock worker A (legacy slot `bf16`) | `00dfc1a3-4d2f-45c2-8585-ebe359a04571` | CIQ-CHI-SPARK04 | 1 GPU, 8 CPU cores, 80 GiB requested memory |
| Stock worker B (legacy slot `fp8`) | `b1b2876e-4ecd-40d8-8574-9ea9fb9bf30c` | CIQ-CHI-SPARK03 | 1 GPU, 8 CPU cores, 80 GiB requested memory |
| Gateway | `4a8b8149-fb63-4dfa-9890-84736bcd3b1c` | CIQ-CHI-SPARK04 | 1 CPU core, 2 GiB, no GPU |

## Same checkpoint, two stock workers

Both API model IDs are `nvidia/Qwen3.8-27B-NVFP4`. Both engines use vLLM
0.28.0 and revision `482ca0f3832238542f8f5295dde86b5f22711d80`.
The existing `qwen38-27b-shared` persistent volume is mounted at `/models`;
the checkpoint directory is
`/models/nvidia--Qwen3.8-27B-NVFP4/482ca0f3832238542f8f5295dde86b5f22711d80`.
Its existing `VERIFIED.json` records 21,945,291,730 bytes, 19 files and
completion/checksums. No new download was needed for this replacement.

Weights retain NVIDIA's mixed NVFP4/FP8 checkpoint format. The stock workers
omit explicit compute/KV dtype overrides. Their startup logs resolve compute
to `torch.bfloat16` and KV configuration to `auto`; they are not FP16 workers.
No FP16 inference test completed before the explicit FP16 pair was canceled.

The gateway's existing `bf16` and `fp8` names are now legacy route-slot labels,
not descriptions of the workers' actual precision. Keeping those labels,
addresses and port preserves the gateway process and its session map.

Common settings: 8,192-token context; maximum 4 sequences and 2,048 batched
tokens per scheduler iteration; 8 GiB cache allocation; tensor/pipeline
parallelism 1; eager execution; text-only mode; prefix caching and chunked
prefill enabled; no speculative decoding. Explicit cache bytes take precedence
over GPU memory utilization. Both replacement workers select FLASH_ATTN.
Functional smoke tests do not establish accuracy or comparative performance.

## Connect and test from the Mac

API base URL:

```text
https://endpoint-f2584b6a68a012358ec3b09dce917a280fd7d082.endpoints.fb-hack.civilfritz.net/v1
```

This is a Fuzzball user-scoped endpoint. Refresh its two-hour token using the
existing Fuzzball login, from the repo root:

```sh
.venv-precisionsplit/bin/python deploy/fuzzball/authorize_endpoint.py \
  --endpoint-id endpoint-f2584b6a68a012358ec3b09dce917a280fd7d082 \
  --output .local-inference/precision-split/greedy-gateway-token
```

The token file has mode 0600. The approved cluster CA is at
`.local-inference/precision-split/cluster-ca.crt`. Both files are ignored by
Git; TLS verification remains enabled. API clients send an Authorization
bearer token. Use the team's approved channel for credentials on another
machine; credential values do not belong in chat, source, or a manifest.

```sh
.venv-precisionsplit/bin/python scripts/test_precision_split_gpu.py \
  --url https://endpoint-f2584b6a68a012358ec3b09dce917a280fd7d082.endpoints.fb-hack.civilfritz.net \
  --token-file .local-inference/precision-split/greedy-gateway-token \
  --ca-file .local-inference/precision-split/cluster-ca.crt \
  --disable-thinking \
  --output results/greedy-gateway-2026-11-15/mac-smoke.json
```

The test sends two concurrent 64-token-capped requests, an 8-token follow-up
and an 8-token streaming request. It checks both workers, session affinity,
a completed stream and zero remaining reservations. Thinking is disabled
for these short tests through `chat_template_kwargs.enable_thinking=false`.
Otherwise it is a request-level choice; a tiny output cap can be consumed by
reasoning before an answer appears. This test is not a performance study.

## Routing on the greedy-gateway branch

This section records the policy in the existing running workflow. The current
repository's `vllm-gateway` now uses the saved E2E predictor on its normal
request path; see [predictive routing](../../precision_split/PREDICTIVE_ROUTING.md).
The bundle has been validated, but that source change has not been rolled into
the running Fuzzball service or its existing public URL.

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

The live gateway embeds this branch's least-outstanding-request implementation.
Local edits require a new gateway deployment to take effect.

## API, configuration and deployment

- `GET /health`: both backend health checks; 200 when at least one is healthy.
- `GET /state`: reservations, session pins, health, model and estimate source.
- `GET /v1/models`: exact served model discovery.
- `POST /v1/chat/completions`: messages, tools and streaming.

Supply `X-Session-ID` for affinity or reuse its returned value. Responses
also include `X-PrecisionSplit-Worker` (`bf16` or `fp8`) and
`X-PrecisionSplit-Request-ID`.

[precision-split-qwen38.json](precision-split-qwen38.json) configures
`http://192.168.99.13:18027` (stock A, legacy `bf16` slot) and
`http://192.168.99.12:18027` (stock B, legacy `fp8` slot).
The gateway listens on 19038. These private node addresses must be checked
and updated after rescheduling. From the Mac use the authenticated HTTPS URL.

[qwen38-fp16-kv.fz](qwen38-fp16-kv.fz) records the canceled explicit FP16
attempts; it does not reproduce the current stock workers. The BF16 and FP8
Fuzzfiles also remain historical examples.
Each requests one GPU. Placement put them on separate nodes; these files
do not contain unique node-affinity labels. Do not start duplicate services
while current allocations are running.

```sh
.venv-precisionsplit/bin/python deploy/fuzzball/render_precision_gateway.py \
  --output results/greedy-gateway-2026-11-15/gateway-next.fz
fuzzball workflow validate results/greedy-gateway-2026-11-15/gateway-next.fz
```

Rendering/validation does not update a running service. Replacement needs a
new deployment, port planning and new endpoint credentials. Source/config
are embedded in the Fuzzfile. The CPU gateway reuses the cached vLLM image's
HTTP dependencies without loading a model or requesting a GPU.

See [routing.py](../../precision_split/routing.py) for atomic reservations,
[forwarding.py](../../precision_split/forwarding.py) for HTTP forwarding and
[vllm_gateway.py](../../precision_split/vllm_gateway.py) for the real API.
Earlier BF16/FP8 test records are in `results/greedy-gateway-2026-11-15/`; object-cache copies
use `fb://group/precision-split/4a8b8149-fb63-4dfa-9890-84736bcd3b1c/`
with seven-day TTL. Credentials are excluded.

Superseded Qwen2.5 workflows: `34067792-1bbd-42fe-9c94-11373c9fcef4`
(FP16/auto KV), `6778fb91-dcca-4ca9-b5d0-143a0356cf39` (FP8 KV), and
gateway `56378649-c8fa-4de9-8982-b66dc1fb7fce`.
All three superseded workflows are confirmed canceled. Historical files
`qwen-fp8-kv.fz` and `precision-split-qwen.json` still describe Qwen2.5;
current renderer defaults use Qwen3.8.

The immediately preceding Qwen3.8 workflows
`52626fae-3879-4559-85c0-d2de4a124f96` (BF16 KV) and
`346f74e3-9826-45dc-adab-6bc95067fcc4` (FP8 KV) were canceled for this
replacement. Their test records remain in `results/greedy-gateway-2026-11-15/`.

Canceled FP16 attempts: `b0245156-566a-4796-b5cc-1207514b4c95` and
`cb9eae51-9223-4efc-a3c3-21cd49128559`. Both report `cancelled by user`.
