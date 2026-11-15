# Predictive gateway plan

## Routing boundary

The production rule is **route once, before the first request of a session**.
The gateway assigns a new `X-Session-ID` to one worker, records the decision,
and pins all later turns in that session to it.  It never switches an in-flight
stream and never silently moves a normal follow-up turn.  A failed pinned worker
returns an explicit upstream error; it is not retried on the other worker.

This keeps the local KV context useful and satisfies the cache-affinity design.
Moving a request mid-generation is not an ordinary OpenAI-compatible gateway
operation. vLLM does document an experimental disaggregated prefill/decode mode
with an explicit KV-transfer connector; it requires a different serving topology,
protocol, and correctness/load validation. It is out of scope for tomorrow's
two full serving engines.

## Four experiments

1. Two stock workers, gateway baseline `least_queue`.
2. Two verified FP8-KV workers, gateway baseline `least_queue`.
3. Two verified BF16-KV workers, gateway baseline `least_queue`.
4. One verified FP8-KV and one verified BF16-KV worker. Compare the sticky
   `least_queue` baseline to the predictive policy on identical held-out,
   multi-turn traces.

Each scenario needs its own worker manifests, launch-log evidence, cold/warm
cache procedure, identical model/tokenizer/sampling limits, and repeated
offered-load cells. Never infer KV dtype from the checkpoint weight format.

## Model and decision rule

For every eligible worker, the router forms a feature vector at dispatch from:

- pinned-tokenizer prompt input tokens and requested output cap;
- request mode flags and previous-turn count;
- worker configuration identity/precision;
- latest *completed-before-dispatch* worker metrics: running requests, waiting
  requests, KV occupancy and state-sample age; and
- gateway-local outstanding count.

It predicts TTFT and E2E separately per candidate worker. The first production
candidate policy is: reject unhealthy/stale/overload candidates; choose the
eligible worker with the lowest predicted E2E; use predicted TTFT as an SLO
gate; then fall back to least-outstanding (or the agreed `least_queue` baseline)
when prediction, model compatibility, or in-domain checks fail. The decision log
stores every candidate feature vector, prediction, chosen worker, fallback
reason, dispatch, and outcome.

`output_tokens`, observed server prompt/cached tokens, final latency, errors,
and quality grades are outcomes. They are retained for training/evaluation but
are never online features. A separate quality gate determines whether FP8 is
eligible at all; do not trade quality away through a latency score.

## Today's seed dataset

Run `build_predictive_router_dataset.py` over the BF16 and FP8 measurement
directories. It uses every selected request as one per-worker training row and
creates a second matched-pairs file keyed by `logical_request_id`. The pair file
is for direct FP8/BF16 diagnostics, not fabricated counterfactual labels:
the two requests were separate measurements at different snapshots.

The 200 successful long-context observations validate ingestion, grouping, and
portable-model export. They are not adequate to promote a router: one workload
family, one output cap, one repetition, no overloaded/error variation, and no
real gateway path. Keep them as seed data and include them in the eventual
campaign only if the worker configuration hashes still match.

## Training and promotion sequence

1. Collect the four scenarios over multiple prompt lengths, output caps,
   arrival patterns, concurrency/offered-rate cells, and multi-turn sessions.
   Repeat cells and keep complete session groups in one split.
2. Use the dataset builder for all raw `requests.jsonl`/`trace.jsonl` runs.
   Preserve failures and nulls; train latency only on completed labels but train
   success/SLO separately when there is outcome variation.
3. Train `train_regression.py` with the grouped train/validation/test rows.
   Inspect held-out MAE/Brier error against constant and `least_queue` baselines,
   by precision, load, context length, and session turn.
4. Shadow the predictive policy in the live gateway: log its proposed choice but
   actually execute `least_queue`. Promote only if offline and shadow checks are
   acceptable and the model has no configuration/feature-domain abstentions.
5. Run matched held-out gateway experiments for scenario 4. Report p50/p95/p99,
   throughput, SLO goodput, failures, quality, fallback rate, and gateway
   overhead. A mixed-precision result alone does not establish a learned-routing
   improvement.

## Gateway integration contract

The gateway owner should call a pure policy adapter only for an unassigned
session. Inputs are the candidate snapshots and request metadata above. Output
is `{worker_id, policy, predictions, fallback_reason}`. The adapter must not
perform HTTP forwarding, mutate vLLM state, change an existing assignment, or
consume post-dispatch fields. Retain the current sticky least-queue behavior as
the always-available fallback.
