# Scenario 4: predictive mixed-KV gateway benchmark

This directory contains the completed held-out gateway-path run from
November 15, 2026. It replays the same frozen 144-request VAST suite used by
the stock gateway benchmark: six 24-request blocks (`c1-r1` through `c4-r3`),
with closed-loop concurrency 1 and 4 and three repetitions per level.

## Verified serving configuration

- Gateway workflow `49774fbc-3324-4b23-b9f7-f5bf46a17db9` used policy
  `minimum_predicted_completion_time` and loaded prediction-model version
  `2a9372b996c76518158cd091e1fba9366d265bcf3d8791e0039108eee05c89d5` with
  no model-load error.
- BF16 worker workflow `814f4652-f73c-4cd2-99d5-a141e7bf8c61` used BF16 model
  and BF16 attention KV cache on `192.168.99.13:18027`.
- FP8 worker workflow `50fbe486-0cc2-42d0-906a-3db329f0451d` used BF16 model
  and resolved FP8 E4M3FN attention KV cache on `192.168.99.12:18027`.

`manifests/gateway.json` and `gateway-state-after.json` preserve the exact
gateway configuration and post-run health. No benchmark-client warm-up
generations were included because the live engines had already completed their
engine/JIT validation before this run.

## Contents and use

Each block includes raw request, response, telemetry, dispatch-ledger and
canonical records. Each canonical record is one attempted request and includes
the request/session ID, tokenizer-based input count, output limit, timestamps,
TTFT, end-to-end latency, actual output count, status, and the observed
`gateway_routed_worker` response header.

All 144 requests succeeded. The observed split was 39 BF16 and 105 FP8
assignments. The gateway does not expose a per-worker metrics snapshot through
this collection path, so `running_requests_at_dispatch` and
`waiting_requests_at_dispatch` are correctly represented as missing rather
than inferred. Do not treat the aggregate p50/p95 or the routing split as a
claim that the predictive policy outperforms the stock baseline: compare the
matched scenario-1 records and quality outputs before making that claim.
