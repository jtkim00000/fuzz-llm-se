# 3. Train the regression model and benchmark two workers

## Goal

Use the single-worker measurements from steps 1 and 2 to build a cost predictor,
then evaluate the gateway with two simultaneously running Nemotron workers on
two GB10 machines. Model training, gateway, worker telemetry, trace replayer
and plotting are planned components; this repo does not yet implement them.

## 1. Build a training dataset

Join verified step 2 measurements by run/session and worker configuration;
preserve raw records and manifests. Use features known when a session is
assigned: estimated input and cached context, output budget or predicted
output length, precision, worker queue/active state, and available cache
occupancy. Use actual completion time or server prefill/decode timing as
targets where measured. Do not use actual future output length, future queue
state, or the other worker's unobserved live result as features.

Collect loaded, multi-turn data in addition to isolated samples: batching and
queueing affect the prediction. A fixed trace replayed separately on each
precision provides useful paired comparisons, but changing queue state means
it is not an exact live counterfactual. Record missing measurements as missing.
Split by complete sessions/traces and preferably separate runs; reserve a
separate final benchmark set. Avoid leaking follow-up turns across splits.

Start with a lookup table or simple regression for prefill/decode or total
cost. Train the chosen regression model only after selecting its features and
units; save model version, dataset hashes, feature schema and configuration.
Evaluate on held-out data with errors by context length, cache state and load,
plus routing choice and baseline comparisons. A queue estimate plus isolated
service estimate is an approximation: continuous batching couples requests.
Use uncertainty or out-of-range fallback and validate actual loaded behavior.
Do not promise an oracle outcome or accuracy from limited data.

## 2. Prepare the dual-worker system

Fuzzball runs one persistent worker on each ThinkStation. Confirm both use the
same Nemotron checkpoint, pinned vLLM build, tokenizer, sampling, context and
comparable GPU allocations. The gateway exposes an OpenAI-compatible streaming
endpoint, chooses among healthy workers for new sessions, maintains session
affinity, records decisions, and uses the heuristic if the model is unavailable.
Use worker metrics for live queue state; do not confuse Fuzzball's job queue
with the inference worker's request queue. Define any correctness constraints
before the final run. Preserve the stock settings from step 1 precisely.

## 3. Compare four configurations

| Configuration | Worker A | Worker B | Routing |
| --- | --- | --- | --- |
| Two standard | Stock settings from step 1 | Same stock settings | Session-sticky baseline |
| Two BF16 | Explicit BF16 KV | Explicit BF16 KV | Same baseline |
| Two FP8 | Explicit FP8 KV | Explicit FP8 KV | Same baseline |
| Mixed | Explicit BF16 KV | Explicit FP8 KV | Heuristic and learned policy in separate runs |

The first three use stock vLLM as an engine, with different KV settings. If
stock resolves to BF16 or FP8 and every other setting is identical, that pair
of configurations is effectively a duplicate; report this instead of
claiming four distinct precision choices. If custom kernels are involved,
compare them as an additional controlled factor, not silently inside one row.

Each configuration requires real Fuzzball workflows and reachable endpoints;
the previous single-worker AIPerf command does not implement the gateway or
multi-turn load generation. Point AIPerf at the gateway after confirming its
API and session header contract. Replay the same held-out fixed traces and
arrival schedules for all configurations, including cold and warm sessions.
Ensure every baseline uses session affinity. Test low, medium and near
saturation offered rates; retain failures/timeouts and actual arrival rates.
Run at least three repetitions of the headline case. Use the same gateway path
and hardware allocation across rows where feasible; document differences.

Record AIPerf's client metrics, worker-side metrics, per-request decisions,
predicted and actual latency, actual token counts and cache hits, errors, and
server/worker state where available. Measure direct-to-worker versus gateway-
to-the-same-worker with matched workload to estimate gateway overhead. Check
that the load generator and gateway are not the bottleneck. Run correctness
tasks for each precision and a routed sample using a predeclared threshold.

## 4. Analyze and present

- Plot p95 end-to-end latency and p95 TTFT per configuration with sample
  counts and variation across repeated runs; report failures and timeouts.
- Plot throughput and latency versus offered load and separate cold from warm
  sessions. Record actual output counts and cache reuse.
- Plot predicted versus observed completion time and errors on held-out runs,
  grouped by worker and load; compare the learned policy to the heuristic.
- Plot correctness pass rates by task category and precision with sample sizes.
- Compare each worker to the matched single-worker measurements in step 2;
  isolate differences in network, concurrency, cache and gateway overhead.

Report whether mixed routing beats both homogeneous baselines on the same
trace. A tie, regression, insufficient data or unsupported mode is a valid
result. Preserve raw JSONL, summaries, traces and hashes, model artifact,
hardware/engine/worker configuration, provenance and errors in an approved
shared location. No benchmark result is established until the real run.
