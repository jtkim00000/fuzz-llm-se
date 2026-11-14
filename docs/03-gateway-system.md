# 3. Profile the gateway and combined system

## Goal and prerequisites

Evaluate whether session-aware routing across the BF16 and FP8 workers improves
the system over strong baselines. The gateway, routing model and worker
telemetry are not yet implemented in this repo. Finish step 2 and first verify
that the gateway can stream requests, identify workers and keep follow-up turns
on the same worker. Fuzzball manages the persistent workers; the gateway makes
the session placement decision.

## Collect training data

Use step 2's paired direct-worker runs for basic cost estimates. Collect
additional traces under low, medium and near-saturation offered load; capture
queue and active-work state **at assignment**, along with session ID, input,
cached tokens when observable, output budget, precision and selected worker.
Record actual response times, actual output tokens, errors and predictions.
The true output length and future worker state are not available at decision
time. Continuous batching changes execution time under load, so isolated
service time plus queue delay is an approximation to validate, not a guarantee.

Split by whole sessions/traces and runs for training, validation and testing;
keep separate final evaluation traces. Record the router model version and
feature definitions. A routed request only reveals the outcome on its chosen
worker; it does not reveal what would have happened on the other worker under
the same changing load.

## Evaluate the complete system

Run the same fixed, held-out session traces, arrival rates, model settings and
cache procedures through each configuration:

1. Two BF16 workers with a session-sticky baseline policy.
2. Two FP8 workers with the same baseline policy.
3. Mixed BF16/FP8 with a session-sticky heuristic.
4. Mixed BF16/FP8 with the trained router.

Keep the same gateway code path for all configurations where feasible, and
document differences. Repeat the headline comparison at least three times.
Capture end-to-end client timing, routing decisions, worker state, cache
reuse, errors and task correctness. If changing precision requires restarting
workers, document restart, warmup and cache isolation. For gateway overhead,
compare a direct worker run and a gateway-to-the-same-worker run under matched
load; network and cache differences must be controlled.

Replay realistic cold analysis and warm multi-turn sessions. Use request-rate
or trace replay to study queues, rather than relying only on fixed concurrency.
The example single-worker command in the [GB10 runbook](GB10-RUNBOOK.md) is a
starting connectivity check, not an implementation of this experiment.

## Graph and compare

- Headline p95 end-to-end latency by configuration on the same mixed trace,
  with the number of requests, repetitions and variation shown. Collect enough
  requests for a meaningful p95; do not infer it from the five-request smoke
  test.
- Latency versus offered load and completed requests/s; include failures and
  timeouts so overloaded runs cannot appear artificially fast.
- Separate cold and warm workload results, cache reuse and routing decisions.
- Predicted versus actual completion time on held-out sessions and prediction
  error by load. Compare learned routing with the heuristic and homogeneous
  baselines, not only with an offline oracle.
- Correctness pass rates by category and configuration, including sample sizes
  and the predeclared threshold. Show latency impact of any BF16-only rule.
- Compare gateway-mediated worker performance with the **matched direct worker
  measurements from step 2** to understand overhead and load effects; do not
  attribute differences to routing when the traces or settings differ.

Preserve raw AIPerf outputs, server metrics, trace hashes and a manifest as
described in the runbook. Store generated artifacts outside Git in team-approved
storage. Report a tie or regression honestly; no speedup is assumed.
