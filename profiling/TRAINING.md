# Collecting load-aware data and training the first router model

This supplements the AIPerf baseline. `run_baseline.py` still provides AIPerf's
standard measurement exports. `collect_training.py` is an additional load driver
that logs all candidate-worker states before dispatch and associates them with
observed response outcomes. It can replay direct single-worker or two-worker
routing experiments. It is a benchmark router, not the production HTTP gateway.
No real model results or trained deployment-quality model exist yet.

Current verified deployment:
- Tools: `/results/baseline-prep-qzFEdMaQ/tools`.
- Client interpreter: `/results/benchmark-env-xzAhY13Y/venv/bin/python`.
- AIPerf 0.13.0 and scikit-learn 1.7.2 installed.
- Final tool deployment workflow: `ec10eb39-487a-4466-9e8a-cf10968ce346`.
- Training dependency check: `/results/training-dependency-check-2IhIFiYQ`.
- Preview inputs are still character-sized; generate the tokenizer-sized training
  suites before collection. The old baseline root is not the extended tool root.


## Operational sequence

1. Obtain the deployment handoff in TEAM-HANDOFF.md. Fill a worker manifest for
   each actual configuration, including exact `launch_arguments`. Verify effective
   dtype, image/model revisions and tokenizer. Unknown metrics remain null.
2. Reuse the verified tools above, or redeploy future changes: run `python3 profiling/make_prepare_workflow.py`,
   validate and submit the generated YAML, then record its NEW prepared root.
   Existing `/results/baseline-prep-IQ1zkCdX/tools` predates these additions.
3. Install training dependencies in a compatible client environment. The updated
   `workflows/install-client.yaml` installs AIPerf 0.13.0 and scikit-learn 1.7.2.
   The existing environment listed above has already been upgraded. A different
   AIPerf-only environment must be upgraded with
   `python -m pip install -r requirements-training.txt` using its own interpreter,
   or replaced through the updated install workflow. Capture the resolved lock file.
4. Generate finalized suites with the pinned local tokenizer. Begin at 8K, then
   add supported lengths such as 16K/32K/65K. Choose identical `--seed` and
   `--namespace` across lengths for matched session groups. For a separate final
   benchmark corpus use a new namespace, not one used for fitting/tuning.
5. Run the AIPerf pilot and accuracy controls before loaded data collection.
6. Build fixed arrival traces; fill collection config and campaign JSON; run
   single-worker sweeps on stock/BF16/FP8 separately. Launch/switch workers with
   the serving team: a configuration label never changes actual worker precision.
7. Export candidate rows, train the forest, and inspect held-out predictions and
   errors. Compare mean-prediction and heuristic baselines. Keep entire sessions
   together. A model artifact existing does not mean its accuracy is acceptable.
8. Run two-worker configurations with round-robin, least-outstanding, least-queue,
   then regression policies on identical held-out traces. Session affinity is
   enforced. Health/overload rules can reject a session; no cache migration is
   silently performed. Production gateway integration is a separate deployment.

These commands run INSIDE the CPU job, with paths replaced by real mounted paths.
`TOOLS` refers to the newly deployed scripts and `PYTHON` to the installed client
interpreter. Both are placeholders below, not environment variables supplied
implicitly by Fuzzball.

```sh
PYTHON=/results/benchmark-env-xzAhY13Y/venv/bin/python
TOOLS=/results/baseline-prep-qzFEdMaQ/tools

$PYTHON "$TOOLS/build_suite.py" --out /results/inputs/reactome-8k-v2 \
  --tokenizer /PINNED/TOKENIZER --tokens 8192 --count 300 --seed 42 --namespace calibration-v2
$PYTHON "$TOOLS/selftest.py" /results/inputs/reactome-8k-v2

$PYTHON "$TOOLS/build_training_trace.py" --suites /results/inputs/reactome-8k-v2 \
  --out /results/traces/short-train-v2.jsonl --split train --kind short --arrival concurrency \
  --slo-ttft-ms YOUR_TTFT_TARGET --slo-e2e-ms YOUR_E2E_TARGET

$PYTHON "$TOOLS/run_campaign.py" --spec /results/filled-campaign.json \
  --out /results/campaign-UNIQUE_ID

$PYTHON "$TOOLS/train_regression.py" \
  --data /results/campaign-UNIQUE_ID/training-export/training.jsonl \
  --out /results/models/UNIQUE_MODEL_VERSION
```

Create corresponding `--split validation` and `--split test` files at distinct
paths and collect each in separate runs. Add all three to the campaign cells.
The trainer rejects rows from the same run assigned to different splits. This
prevents shared bursts/batches from leaking between evaluation and fitting.

Trace variants: `--kind long` or `accuracy`; `--arrival constant`, `poisson` or
`bursty` with `--rate REQUESTS_PER_SECOND`; `--multi-turn` for two live turns on
performance traces. Bursts are 10 sessions spaced by `10/rate` seconds. Subsequent
turns arrive after previous completion plus explicit think delay. Live generated
history can differ between precision runs. Per-turn trace objects can also specify priority, SLO thresholds, `tools`,
`response_format`, `chat_template_kwargs`, and prior `expected_output_tokens`.
The collector derives tool/structured-output flags from actual request fields;
only send modes supported by the actual model server.
For a strictly fixed-history trace,
provide complete `messages` per turn manually. Trace groups are blocks of 10 source sessions, kept stable across lengths and
arrival patterns, and carry train/validation/test assignments (70/15/15 hash rule); repeated cells and matching
length variants keep the same group split. Do not random-split exported rows.
A burst uses 10 sessions, so the split group keeps that burst intact.
Generate at least 300 source sessions for the starter training campaign and inspect
actual split counts; 100 inputs remain adequate only for the initial baseline.
Generate enough independent groups in all splits; the trainer refuses fewer than
3 per split. This minimum is a pipeline check, not evidence of statistical power.

## Collection configuration

Copy `collection-config.template.json` onto writable results storage. Fill worker
base URLs (no `/v1` suffix), model IDs, manifests, metrics and health endpoints.
For two workers add a second entry with unique `id`. Supported policies:
`single`, `round_robin`, `least_outstanding`, `least_queue`, `regression`.
All policies maintain affinity; all-overloaded/unhealthy conditions are recorded
as explicit rejected requests, not silently sent to an arbitrary worker.

Optional `model_path` points to the trained JSON forest. Regression falls back to
least-outstanding if no artifact, an unseen worker configuration, out-of-range
features or unavailable predictions prevent a valid comparison. Predictions
include queue effects already; do not add queue delay again without measuring the
result. Current selection minimizes predicted E2E; SLO and reliability models are
reported separately. The coefficients proposed in earlier design text are not
hard-coded. Quality approval remains a separate worker eligibility constraint.

`limits` allows KV/queue thresholds and `require_quality_approval`. Configure
thresholds from your observations and latency requirements. If a configured
threshold's metric is missing/stale, the candidate is excluded. Unknown health
without a configured health endpoint is not asserted healthy. No migration on a
failed affinity worker; the session is rejected and recorded. The benchmark logs
zero migrations by construction. Do not enable FP8 quality approval until graded
measurements meet a predeclared tolerance.

`metric_map` is explicit and must be checked against the pinned engine's actual
Prometheus names/labels. Mapping rules accept `metric`, exact `labels`, aggregation
`sum/mean/min/max`, and `scale`. Gauges are sampled; counters require
`"kind":"counter_rate"` to become rates. Counter resets yield missing samples.
Choose per-worker label filters if an endpoint contains multiple replicas. Never
interpret memory allocated as memory bandwidth or scrape round-trip time as pure
network RTT. vLLM metric names vary: templates are candidate mappings, not proof
of availability. `running_requests` is deliberately separate from unknown running
sequence counts. Raw matching samples and their labels are retained.

For GPU/CPU/memory exporter endpoints add worker `metric_sources`, for example:

```json
{
  "metric_sources": [{
    "url": "http://ACTUAL_EXPORTER:9400/metrics",
    "metric_map": {
      "gpu_utilization": {
        "metric": "DCGM_FI_DEV_GPU_UTIL",
        "labels": {"gpu": "0"},
        "aggregation": "mean",
        "scale": 0.01
      }
    }
  }]
}
```

This is a mapping example, not a deployed GB10 exporter. Confirm exporter support
and units on the venue hardware. All other hardware measurements stay null until
configured. Optional per-response `response_label_paths` can extract verified
server-specific timing fields, e.g. a configured queue-time JSON path and unit
scale. Do not map aggregate server histogram quantiles to an individual request's
queue/prefill/decode labels. Unsupported metrics are disclosed as missing.

Credentials are read from each worker's `api_key_env` (default
`BENCHMARK_API_KEY`); inject through the venue's secret mechanism. Only env names,
not values, are saved. URLs must not embed credentials or query secrets. The
additional streaming collector supports authenticated HTTP independently of the
older AIPerf wrapper's still-pending venue auth integration.

## What gets measured and saved

- `telemetry.jsonl`: timestamped source observations, health, scrape errors,
  explicit missing metrics and scrape durations.
- `requests.jsonl`: request/session/group IDs, split, arrival/decision/dispatch/
  completion times, all candidate snapshots with sample ages, pre-dispatch
  features, selected worker/reason/prediction, affinity, and observed labels.
- `responses.jsonl`: raw JSON events, timestamped content/reasoning chunks,
  model outputs. Usage supplies actual prompt/output/cached counts when exposed.
- `run-config.json` and `provenance.json`: effective configuration manifests,
  trace/code hashes, tokenizer location, client identity and cache procedure.
- `summary.json`: throughput, TTFT/E2E percentiles, SLO attainment/goodput,
  failures and client schedule delay. Accuracy is separate from performance.
- `training-export/training.jsonl`: one candidate row per decision; only the
  selected candidate has outcome labels. Replays are not exact counterfactuals.
- Model outputs: portable `model.json`, `evaluation.json`, held-out predictions
  with context/load columns for additional plots and analysis.

TTFT includes network/queue/server overhead. First-visible-content timing is
separate from first reasoning/output timing. SSE chunks may contain multiple
tokens: exact ITL percentiles remain NULL unless supplied by verified per-request
server instrumentation. Chunk-gap quantiles and approximate TPOT from actual
usage are named separately. Use AIPerf for its defined ITL metrics and compare the
client measurements on matched low-load pilots. No unverified AIPerf/request-ID
join is attempted. Server phase timings, bandwidth and pure network overhead
cannot be fabricated from client latencies.

The periodic collector logs latest completed snapshots BEFORE each decision,
including timestamps, sample age, and sequential-source sampling skew. It is not
an atomic GPU-state measurement. Set a reasonable sampling interval/age policy;
sampling itself adds traffic. Candidate state is copied before dispatch, not
reconstructed from future metrics. Per-request `estimated_cached_tokens` can be supplied only if known before routing;
otherwise it stays null, separate from observed cached tokens returned afterward.
Client outstanding counts are identified as
client observations and do not include other traffic.

Open-loop arrivals use a bounded client thread pool and retain client queue delay.
`arrival_to_completion_ms` includes that delay; E2E regression labels measure
actual HTTP dispatch through completion. Inspect client delay to detect a client
bottleneck; move/scale the generator rather than claiming this delay is a vLLM
queue. SLO thresholds currently apply to dispatch-based latency, not scheduled
arrival-based latency. The collector stores no universal per-token timestamp
truth. Tool/structured-output/thinking flags must match actual request settings.

## Campaigns, training and interpretation

`campaign.template.json` specifies configuration files, traces, concurrency cells
and repetitions. Add cases for lengths and offered-rate patterns. Automatic stop
conditions check observed errors, client delay and optional SLO attainment after
each run. The baseline template thresholds are starter guardrails, not hardware
recommendations. State-based KV/queue exclusion occurs at each decision when
configured. Campaigns do not restart serving engines or auto-clear caches: agree
and record cold/warm/restart procedures before matched repetitions. A reused
trace may hit cache; the default cache condition is explicitly unverified.

Set `train_after_collection` to true to train automatically after export, only
once worker manifests are verified and all splits have sufficient groups. By
default collection/export completes and training is a separate explicit command.
Never silently discard failed requests to make reliability results look better.
Session follow-ups not attempted after a failed earlier turn are counted separately.

The trainer uses an explicit pre-dispatch feature allowlist and trains forests
for TTFT/E2E, plus SLO and success classifiers when labels are present. Medians and
missing indicators are fitted on TRAIN only; validation/test remain separate.
Constant-label reliability data yields a clearly labeled constant estimate, not
proof of reliability. Probabilities are uncalibrated. Failed HTTP requests are
excluded from latency fitting and retained for success/SLO classifiers. Actual
output length, cached tokens after completion and other future measurements
remain labels, never online features. Timing labels do not absorb accuracy labels.

Default training refuses unverified configuration rows. `--allow-unverified` is
for explicit mock/pipeline checks only. Models preserve training data hashes,
feature schema, configuration hashes, units as named in the schema, sklearn
version and seed. Inspect held-out MAE/Brier error versus the constant baseline,
by precision, and raw held-out predictions by context/load. The initial model is
not automatically promoted into a live serving gateway.

Tests: run `python -m unittest discover -s profiling/tests -v` in an environment
with AIPerf dependencies and requirements-training.txt. They cover strict grading,
Prometheus label selection/missingness, affinity/health/overload behavior, a local
mock streaming server, candidate export and grouped training/leakage rejection.
They validate the pipeline, not Nemotron quality or performance.
