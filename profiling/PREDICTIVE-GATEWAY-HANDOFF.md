# Handoff: train and safely deploy the predictive BF16/FP8 gateway

## Goal and routing boundary

Build a gateway policy for the mixed-precision topology: one BF16-KV vLLM worker
and one FP8-KV vLLM worker serving the same model. For a **new session only**,
predict which eligible worker is more likely to meet latency objectives and route
there. Persist that worker assignment by `X-Session-ID` for every later turn.

Do not switch an in-flight stream, silently move a subsequent pinned turn, or
attempt to migrate the KV cache. A mid-generation switch would require a different
vLLM KV-transfer/disaggregated-serving topology and is out of scope. If a pinned
worker fails, return an explicit upstream error and record it; do not retry the
turn at the other worker.

The first safe objective is predicted end-to-end (E2E) latency subject to health,
fresh telemetry, overload, and separately-approved quality gates. TTFT is a
second predicted target and an SLO gate. It is not sufficient to choose the
precision with the best average historical latency.

## What exists now

Repository/branch: `profiling/bf16-fp8-request-records`, commit `1d102e0` or
later. The data and tools are in the repository and also remain on the Fuzzball
`profiling-results` volume.

| Item | Repository path | Fuzzball results-volume path |
| --- | --- | --- |
| Full matching controlled campaign | `profiling/results/vast-full-bf16-fp8-2026-11-15/` | `/results/vast-full-bf16-fp8-2026-11-15/` |
| Data description | `profiling/results/vast-full-bf16-fp8-2026-11-15/README.md` | same suffix under `/results` |
| Data builder | `profiling/build_predictive_router_dataset.py` | copy/deploy this tracked file before use |
| Portable forest trainer | `profiling/train_regression.py` | copy/deploy this tracked file before use |
| Feature allowlist/inference | `profiling/features.py` | copy/deploy this tracked file before use |
| Collector/router benchmark logic | `profiling/collect_training.py` | copy/deploy this tracked file before use |
| Current HTTP gateway baseline | `gateway/app.py` | deploy separately with the gateway |
| Existing architecture/policy rationale | `profiling/PREDICTIVE-GATEWAY-PLAN.md` | n/a |
| Detailed collection/training instructions | `profiling/TRAINING.md` | n/a |

The full campaign has **288 successful controlled observations**: 144 BF16 and
144 FP8. For each precision it has six 24-request blocks: concurrency 1 and 4,
three randomized repetitions each; four formatted input buckets (about 512,
2,048, 4,096, 7,168 tokens); two forced output lengths (32 and 256); and three
distinct VAST Reactome prompts per cell/repetition. Both workers had
`max_model_len=8192`, so 7,168 was the common safe fallback instead of 16,384.
Every request has raw streaming data, a dispatch ledger, pre-dispatch worker
snapshots, 200-ms telemetry, manifest, trace, and a normalized canonical row.

The prior 32-request compact run is only a pilot and is stored separately at
`profiling/results/vast-compact-bf16-fp8-2026-11-15/`. Older 100-prompt data is
preserved in `profiling/results/deprecated-direct-node-bf16-fp8-vast-reactome-c4-100-2026-11-14/`.
Do not quietly pool those into the current training set unless configuration,
prompt, output-control, and cache-condition compatibility are verified.

## Important limitation of the seed data

Use **all 288 full-campaign selected-worker records** to validate ingestion,
model serialization, grouped splitting, and offline prediction. Do not present a
model trained only on this dataset as a production-ready smart router. The dataset
has one VAST workload family, controlled rather than natural output lengths,
limited load states, no true two-worker gateway decisions, and no counterfactual
measurement of both workers at the exact same live queue state.

The matched BF16/FP8 records are useful paired diagnostics, but they are not
counterfactual labels: their requests ran at different times and therefore saw
different queues. Never manufacture a label such as “what BF16 would have taken
for this FP8 dispatch.”

## Data schema: what must be used and what must never be online input

For each candidate worker at decision time, form features only from information
available **before HTTP dispatch**:

1. Request: pinned-tokenizer input-token count, requested output cap, streaming,
   tools/structured-output/thinking flags, priority, SLOs, prior-turn count, and
   any *known* cached-token estimate.
2. Worker identity: verified KV precision/configuration hash, model/tokenizer
   revision, and backend identity. Reject configuration hashes not represented in
   the model artifact.
3. Fresh snapshot: completed-before-dispatch running requests, waiting requests,
   KV occupancy, state age, and only explicitly validated exporter metrics.
4. Gateway-local state: outstanding requests for that candidate.

`profiling/features.py` is the authoritative allowlist. Extend it only by adding
a value that is demonstrably available before dispatch and recorded with a source
and timestamp. Do **not** use actual output tokens, observed prompt/cached-token
usage, TTFT, E2E latency, completion timestamp, future queue samples, response
text, quality grade, or error outcome as an online feature. Those are labels or
diagnostics.

Targets:

- `e2e_ms`: primary regression target.
- `ttft_ms`: secondary regression/SLO target.
- `success` and `slo_met`: classifiers only once there are both positive and
  negative examples. This dataset may yield constant classifiers; that is not a
  usable reliability model.
- Quality: a separate natural-output evaluation and eligibility gate, never a
  latency-model proxy.

## Step 1: reproduce and validate raw data before training

1. Check out the pinned branch and confirm the full result manifests say
   `effective_kv_dtype: bfloat16` for BF16 and `fp8` for FP8. Do not infer either
   from the NVFP4 checkpoint name.
2. Verify all twelve `canonical-requests.jsonl` files: six blocks per engine,
   24 rows per block, `status == "succeeded"`, and `output_tokens` in `{32,256}`.
3. Verify the BF16 and FP8 `traces/manifest.json` source hash and trace hashes
   agree, and that paired prompt IDs match for every condition/repetition.
4. Preserve all raw data and failures. Do not use graph-level outlier exclusion
   for training, summary statistics, or policy evaluation.
5. Record the model/checkpoint, tokenizer revision, vLLM version, KV dtype,
   attention backend, context/batch settings, prefix-cache condition, and client
   location. A changed serving configuration requires a new data version.

## Step 2: build the seed training dataset from all 288 rows

Run in a CPU environment with the repository and training dependencies. The
prepared Fuzzball interpreter was previously:

```sh
PYTHON=/results/benchmark-env-xzAhY13Y/venv/bin/python
REPO=/PATH/TO/fuzz-llm-se
ROOT=/results/vast-full-bf16-fp8-2026-11-15
OUT=/results/router-seed-vast-full-2026-11-15

cd "$REPO/profiling"
"$PYTHON" build_predictive_router_dataset.py \
  --runs \
  "$ROOT/bf16/c1-r1" "$ROOT/bf16/c1-r2" "$ROOT/bf16/c1-r3" \
  "$ROOT/bf16/c4-r1" "$ROOT/bf16/c4-r2" "$ROOT/bf16/c4-r3" \
  "$ROOT/fp8/c1-r1" "$ROOT/fp8/c1-r2" "$ROOT/fp8/c1-r3" \
  "$ROOT/fp8/c4-r1" "$ROOT/fp8/c4-r2" "$ROOT/fp8/c4-r3" \
  --out "$OUT/dataset"
```

Expected outputs:

- `$OUT/dataset/training.jsonl`: one observed per-worker row for every selected
  request. Expected count: 288.
- `$OUT/dataset/matched-pairs.jsonl`: BF16/FP8 outcome pairs keyed by logical
  request ID, for descriptive analysis only.
- `$OUT/dataset/manifest.json`: source hashes, split counts, feature allowlist,
  and explicit non-feature labels.

The builder uses a stable hash of the logical prompt group for 70/15/15
train/validation/test splitting. Inspect the manifest and reject any attempt to
place a logical prompt/session group in more than one split. Keep all future
turns of a real session together as well.

## Step 3: train, inspect, and version the seed model

Install the pinned training dependency if absent:

```sh
"$PYTHON" -m pip install -r "$REPO/profiling/requirements-training.txt"
```

Then train the existing portable random-forest implementation:

```sh
cd "$REPO/profiling"
"$PYTHON" train_regression.py \
  --data "$OUT/dataset/training.jsonl" \
  --out "$OUT/model"
```

Do not pass `--allow-unverified` for this campaign: its manifests are verified.
The trainer writes:

- `model.json`: portable tree representation, feature list, imputation medians,
  feature-domain bounds, configuration-hash allowlist, and model metadata.
- `evaluation.json`: train/validation/test row/group counts, held-out MAE for
  TTFT/E2E, constant-baseline comparison, and reliability-model warnings.
- `heldout-predictions.jsonl`: held-out predictions with context/load columns.

Review `evaluation.json` before considering any policy. Require held-out error
to improve on a constant baseline and inspect error separately by precision,
input bucket, output cap, concurrency, and repetition. Because this seed set is
small and structured, a statistically weak or unstable result is expected and
is a valid reason to keep the policy in fallback/shadow mode.

Save an immutable model manifest next to the model containing: Git commit, data
directory and SHA-256, all source run hashes, feature schema version, sklearn
version, training command, split rule, evaluation result, approver, and creation
time. Never overwrite an existing model directory; create a new version.

## Step 4: use all data for diagnostics without data leakage

Use `matched-pairs.jsonl` to make paired plots of FP8 minus BF16 E2E/TTFT for
the same logical prompt, condition, and repetition. Report the two live
snapshots and block/ordering differences alongside each pair. Use canonical
files for scatter plots versus input tokens, queued/running requests, output cap,
and concurrency. Show all points and sample counts.

Do not use the held-out test split to tune forest depth, feature selection,
thresholds, or fallback rules. Tune only using training/validation, then perform
one locked held-out test evaluation. Keep the seed model and its report even if
it fails promotion; a negative result is useful.

## Step 5: collect the data required for a promotable router

The next campaign must go through the actual two-worker gateway path. For each
of the four comparison scenarios, use the same total hardware and held-out
multi-turn arrival traces:

1. Two stock workers with sticky least-queue/least-outstanding baseline.
2. Two FP8-KV workers with the same baseline.
3. Two BF16-KV workers with the same baseline.
4. One FP8 and one BF16 worker: compare sticky baseline versus predictive policy.

For training scenarios, collect multiple independent session groups across prompt
lengths, requested output caps, natural output tasks, concurrency and offered-rate
cells (constant, Poisson, and bursty), queue/KV occupancy ranges, and real
multi-turn sessions. Balance worker exposure deliberately during collection:
least-queue alone may starve a slower candidate and leaves sparse labels. Use
pre-declared safe exploration/round-robin blocks or controlled replay runs; log
the policy. Preserve failures/rejections, and include configuration changes as
new data versions rather than blending them silently.

Run a separate natural-output known-answer quality suite for both precisions.
Only permit FP8 as an eligible candidate when its predeclared quality tolerance
is met. The forced 32/256 outputs in this seed data are synthetic timing controls
and cannot establish answer quality.

## Step 6: build the production policy adapter

`gateway/app.py` currently implements only sticky least-outstanding routing. The
next agent must add a small, pure policy module (for example
`gateway/policy.py`) rather than embedding model logic in HTTP forwarding.

Required adapter contract:

```python
decision = choose_new_session(request_features, candidate_snapshots, model)
# returns {
#   "worker_id": "bf16" | "fp8",
#   "policy": "regression" | "least_outstanding" | "rejected",
#   "predictions": {worker_id: {"e2e_ms": ..., "ttft_ms": ...}},
#   "fallback_reason": str | None,
# }
```

For each candidate, the adapter must:

1. Fetch/use only a completed-before-dispatch snapshot and calculate sample age.
2. Reject unhealthy, missing, stale, over-limit, quality-unapproved, or unknown
   configuration candidates.
3. Construct the allowlisted feature vector with `profiling/features.py`
   semantics.
4. Reject the learned path if `in_domain()` fails, any required feature is
   unavailable, the model artifact cannot load/validate, or a prediction is not
   finite.
5. Among eligible model candidates, choose the minimum predicted E2E. Enforce a
   predicted TTFT/SLO gate if configured.
6. Fall back deterministically to sticky `least_outstanding` (or the agreed
   least-queue baseline) when prediction cannot be used.
7. Pin the selected worker before forwarding and never reroute that session.

Log, for every routing decision: request/session ID, candidate feature vectors,
snapshot source/timestamps/age, candidate health/exclusions, model version and
configuration hashes, predictions, selected worker, policy/fallback reason,
dispatch/completion times, worker response/error, and quality result when later
available. Redact credentials and never log response content unless a separately
approved evaluation run requires it.

## Step 7: shadow, then promote only with held-out proof

1. Deploy the model adapter in **shadow mode**: calculate/log its proposed route
   while the gateway actually uses sticky least-queue/least-outstanding.
2. Compare shadow recommendations against baseline on new held-out traffic:
   fallback rate, feature-domain abstentions, prediction calibration/MAE, queue
   state, TTFT/E2E, errors, quality, and gateway overhead.
3. Fix the final model/artifact and thresholds before the one promotion test; do
   not tune on that test trace.
4. Replay the identical held-out multi-turn traces through scenario 4 using both
   baseline and predictive policies. Report p50/p95/p99 latency, throughput,
   SLO goodput, errors, quality, actual routing split, fallback rate, and gateway
   overhead.
5. Promote only if the learned policy improves the predeclared metric without
   worsening quality/error/SLO criteria. Otherwise retain the baseline and report
   the negative result.

## Definition of done for the next agent

Do not call the gateway “smart” or claim a precision benefit until all are true:

- seed dataset/model is reproducible and versioned;
- no online feature depends on post-dispatch outcome information;
- model configuration/feature-domain checks and deterministic fallback work;
- session affinity is tested, including an assigned-worker failure;
- quality gate is independently evaluated;
- shadow logs are reviewed; and
- matched held-out two-worker baseline-versus-learned results meet predeclared
  latency, SLO, reliability, and quality criteria.
