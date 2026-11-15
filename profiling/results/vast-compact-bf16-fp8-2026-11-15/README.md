# Matched direct-node VAST compact campaign (2026-11-15)

This is the completed direct-worker campaign, not a gateway benchmark. It keeps
the older single-node artifacts untouched under
`profiling/results/deprecated-direct-node-bf16-fp8-vast-reactome-c4-100-2026-11-14/`.

## Scope

Two independent vLLM engines served the same checkpoint and frozen VAST
Reactome trace. `bf16/` used verified attention-KV `bfloat16` with FLASH_ATTN;
`fp8/` used verified attention-KV `fp8` with FLASHINFER. Both workers had
8,192 maximum context tokens, `max_num_seqs=4`, and prefix caching enabled.
This is a matched measurement of those live configurations, not evidence about
stock vLLM, gateway routing, quality, or all offered loads.

Each engine completed 16 requests: four actual input-length buckets (roughly
512, 2,048, 4,096, and 7,168 tokens), two forced output limits (32 and 256),
and two independently shuffled blocks (`c1`, concurrency 1; `c4`, concurrency
4). The complete input text, nonce, request ordering, VAST source hash, and
trace hashes are in `ENGINE/traces/`. A short and a long unrelated-shape warmup
were run before collection; cache condition is `prefix-reuse-minimized, not
verified cold`.

## Read these files

- `ENGINE/c1/canonical-requests.jsonl` and `ENGINE/c4/canonical-requests.jsonl`:
  one normalized row per request. These are the graph/training-table starting
  point.
- `ENGINE/c*/dispatch-ledger.jsonl`: request IDs, dispatch/completion times,
  and pre-dispatch worker snapshot linkage.
- `ENGINE/c*/responses.jsonl`: raw streaming chunks, including first/last
  visible chunk timings and server usage.
- `ENGINE/c*/telemetry.jsonl`: 0.2-second Prometheus samples during the block.
- `ENGINE/c*/requests.jsonl`, `trace.jsonl`, `run-config.json`, and
  `provenance.json`: raw client records and reproducibility metadata.
- `ENGINE/manifests/worker.json`: verified effective precision/backend and
  controlled engine settings. `preflight/forced-32.sse` verifies the forced
  32-token readiness response.

## Per-request fields and safe analysis

The canonical rows provide: `request_id`, `logical_request_id`, `worker_id`,
`input_tokens`, `requested_output_token_limit`,
`running_requests_at_dispatch`, `waiting_requests_at_dispatch`, `ttft_ms`,
`output_tokens`, `e2e_latency_ms`, dispatch and completion UTC timestamps,
state-sample timestamp/age, HTTP outcome, and token-count provenance. The raw
files retain fields needed for alternate timing reconstructions.

Filter to `status == "succeeded"` and keep `c1` and `c4` separate before
comparing precision. Compare exact input bucket/output-limit cells, report the
small sample count (one request per cell), and do not pool them into a claimed
speedup. For a routing model, only pre-dispatch values may be input features:
input tokens, requested output cap, block/concurrency condition, worker
identity/configuration, and the snapshot of running/waiting work. TTFT,
completion latency, actual output tokens, and any later telemetry are targets
or diagnostics, never online features. Do not split turns from a session across
training and hold-out data.

The campaign has no answer-quality score. Evaluate quality separately on a
matched task suite before concluding FP8 is acceptable.
