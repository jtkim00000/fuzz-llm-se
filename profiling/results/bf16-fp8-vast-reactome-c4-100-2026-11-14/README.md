# Matched BF16 and FP8 VAST Reactome profile

This directory contains two independent direct-vLLM runs completed on 2026-11-14:
`bf16/` targets SPARK03 and `fp8/` targets SPARK04.  Each has 100 attempts at
closed-loop client concurrency 4, using the identical deterministic VAST
Reactome trace (`source_sha256` in each trace manifest).  Prompt inputs are
5,889--6,200 tokens under the pinned Qwen tokenizer/chat template, with a
requested output limit of 128 tokens.

## Start here

Use `bf16/measurement/canonical-requests.jsonl` and
`fp8/measurement/canonical-requests.jsonl`.  Each JSONL line is one attempted
request, never an aggregate or a dropped retry.  Pair engines by
`logical_request_id`; all 100 pairs share the same logical prompt.  Do **not**
join by completion order.

The canonical fields are:

- Inputs available before dispatch: `input_tokens`,
  `requested_output_token_limit`, `running_requests_at_dispatch`, and
  `waiting_requests_at_dispatch`.
- Outcomes: `ttft_ms`, `e2e_latency_ms`, and `output_tokens`.
- Timing/provenance: `dispatch_timestamp_utc`, `completion_timestamp_utc`,
  `state_sample_completed_timestamp_utc`, `state_sample_age_ms`, `worker_id`,
  `status`, `finish_reason`, `error_type`, and `missing_fields`.

Use rows with `status == "succeeded"` and non-null values for a latency plot,
but report all 100 attempts and any null/error counts separately.  Never use
actual `output_tokens`, final latency, or completion-time telemetry as an
online routing feature: they are outcomes.  Suitable first plots are TTFT and
E2E scatterplots versus `input_tokens`, colored by engine, plus p50/p95/p99
latency and completed-request throughput.  The state fields are the only
queue/load observations intended as pre-dispatch routing features.

## Supporting files

- `measurement/summary.json`: run-level performance summary.
- `measurement/canonical-summary.json`: attempt, success, and missing-field
  coverage for the canonical data.
- `measurement/dispatch-ledger.jsonl`: durable pre-request dispatch record.
- `measurement/requests.jsonl`, `responses.jsonl`, and `telemetry.jsonl`:
  raw outcome, stream, and timestamped metric evidence.  Preserve these for
  reconciliation; do not replace canonical values with inferred aggregates.
- `measurement/run-config.json`, `measurement/provenance.json`, and
  `manifests/worker.json`: endpoint configuration and verified effective
  engine settings.
- `vast-reactome-trace.jsonl` and its manifest: workload construction and
  input-token distribution.  `preflight/` is separately labelled and is not
  part of the 100-request performance sample.

## Interpretation limits

The engine manifests verify BF16 attention KV on SPARK03 and FP8 E4M3FN
attention KV on SPARK04.  The model weights are NVFP4/modelopt-mixed in both
runs; that is a different setting from KV precision.  FP8 selected
FlashInfer/XQA while BF16 used FlashAttention, so this is a matched deployed
configuration comparison, not a proof that cache dtype alone caused any
difference.  Server cache state was not resettable from the client and is
recorded as `unverified`.  This long-context, single-turn trace exercises KV
capacity/bandwidth but does not measure multi-turn prefix-cache reuse.  The
strict JSON retrieval grade is diagnostic only; do not claim quality parity
from this performance run without a separately designed and predeclared
quality evaluation.

## Predictive-router seed preparation

`../../build_predictive_router_dataset.py` turns both `measurement/` folders
into 200 per-worker rows plus 100 matched-prompt diagnostic pairs. It keeps
only pre-dispatch values in `features`; outcomes remain labels. The routing
boundary, training sequence, promotion gates, and gateway contract are in
`../../PREDICTIVE-GATEWAY-PLAN.md`.
