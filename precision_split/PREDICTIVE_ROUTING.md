# Predictive routing in the active gateway

`python -m precision_split vllm-gateway --config ...` serves the existing
`POST /v1/chat/completions` handler in `precision_split/vllm_gateway.py`.
New sessions now use minimum predicted completion time by default. This is
the forwarding decision, not a shadow recommendation. API payloads, response
headers, worker URLs, health checks, tokenization, and streaming are preserved.
The local `/infer` mock and the older `gateway/shadow.py` adapter retain their
separate baseline/shadow behavior.

## Model and score

`CompletionPredictor.score` calls the existing
`profiling.features.feature_row` and `profiling.features.predict` functions.
The latter reuses `vector` for the saved feature ordering, median imputation,
and per-feature missing indicators. No model is trained during serving.

The frozen artifact is [model.json](../gateway/models/vast-full-v1/model.json).
It is the full 288-observation campaign's existing 64-tree latency forest,
recovered from the recorded artifact archive, not the deprecated pilot model.
Its original bytes and SHA-256 are preserved:
`2a9372b996c76518158cd091e1fba9366d265bcf3d8791e0039108eee05c89d5`.
[manifest.json](../gateway/models/vast-full-v1/manifest.json) and
[evaluation.json](../gateway/models/vast-full-v1/evaluation.json) preserve
the original provenance and offline results, including seed-only status.

The `e2e_ms` target measures HTTP dispatch through response/stream completion,
including queueing under the observed load. Therefore the score is simply:

```text
score(worker) = predict(features_for_worker, artifact, "e2e_ms")
```

Scores are milliseconds from dispatch to completion, not absolute timestamps.
Do not add a service estimate, queue delay, TTFT, or the old remaining-work
diagnostic to this score. Doing so would count queueing twice.

The artifact expects input-token count, requested output cap, their sum,
previous turns, streaming/tools/structured-output/thinking flags, precision
indicators, gateway outstanding count, state age, running/waiting requests,
and KV occupancy. Input length comes from the existing worker tokenizer;
output length is the requested cap, never observed generation length.
Unknown values retain the original preprocessing's missing-value semantics.

Precision comes from `workers[...].manifest.effective_kv_dtype` (or the direct
`effective_kv_dtype` field), not from a route label. BF16 and FP8 map to the
artifact's `precision_bf16` and `precision_fp8` indicators. The checked-in
cluster configuration records both current stock workers as BF16; its `fp8`
name is a legacy alias. A real mixed pair needs verified per-worker metadata.

## Load and lifecycle

For a new session, health and tokenization run as before. Metrics snapshots
are fetched concurrently from `/metrics` with backend authentication and a
two-second timeout. The default mapping reads vLLM running requests, waiting
requests, and KV-cache occupancy. `metrics_url` and `metric_map` can override
this mapping. Missing or failed metrics remain missing and are imputed by the
existing model; they are not fabricated as idle measurements.

Inside the reservation lock, each candidate's current gateway outstanding
count is supplied to the feature builder. The predictor runs synchronously,
then selection, session pinning, and reservation commit happen before the
lock is released. Concurrent arrivals therefore see earlier assignments even
when exporter metrics lag. Only healthy workers fitting the configured
context limits are candidates. Equal predictions choose alphabetical worker
name, so ties are deterministic.

Follow-ups bypass worker selection and stay pinned, even when another score
would be lower. They add a reservation too. Existing unavailable-pinned-worker
behavior remains 503. Completion, upstream errors, and cancellation release
the reservation. Streaming holds it until closure. There are no cross-worker
retries or session migration.

Assignment logs include both candidate scores, eligibility, numeric feature
vectors, live counts, model hash, domain diagnostics, and selection reason.
No prompts, generated content or credentials are logged. `/state` reports
the configured policy, model hash, load and session map. The old timing fields
remain diagnostics for compatibility; they do not influence prediction.

If the model cannot load, precision is unknown/unsupported, or a prediction is
invalid, routing falls back to the existing healthy least-outstanding policy
and logs the fallback. Requests outside training ranges or configuration
hashes are scored with explicit diagnostic flags rather than rejected. This
is a deliberate active prototype: the seed was trained on forced outputs and
limited load states. Its predictions, extrapolation, and precision quality
are not validated production latency or accuracy guarantees. The old strict
shadow policy's domain/quality gates remain confined to that adapter.

## Run and verify

```sh
uv pip install --python .venv-precisionsplit/bin/python -r requirements-precisionsplit.txt
.venv-precisionsplit/bin/python -m precision_split vllm-gateway \
  --config deploy/fuzzball/precision-split-qwen38.json --port 19038
.venv-precisionsplit/bin/python -m precision_split.verify_predictive \
  --output results/predictive-gateway-http-verification.json
```

The verification uses three real loopback HTTP servers, the production handler
and the actual saved forest. Controlled workers report 512 input tokens and
lagging zero-load metrics. It checks idle FP8 selection, a held FP8 stream
causing a new session to select BF16, sticky follow-up routing, upstream HTTP
500 cleanup, streaming completion and zero final reservations. Unit tests also
cover connection failure, cancellation, unhealthy pinned workers, context
eligibility, atomic concurrent selection and deterministic ties.

The renderer embeds all inference code and the exact artifact bytes in its
Fuzzball bundle. It uses the existing HTTP path and configured worker URLs.
Rendering/validating does not update an already-running service. The live
Fuzzball endpoint was not replaced by these repo edits: the available workflow
update operation changes priority only. Preserving that exact public hostname
requires deployment support to update or rebind its service; starting another
workflow would create a different endpoint. The original live workers and
gateway are left intact. Functional verification above uses the normal API
over controlled HTTP workers, not the live GPU endpoint.
