# Scenario 1: stock-vLLM greedy-gateway benchmark

Completed on 2026-11-15 through the CPU `least_outstanding_requests` gateway.
This result is separate from direct-worker profiling/training data.

## Coverage

- 144/144 successful measured VAST requests.
- Six frozen trace blocks: `c1-r1` through `c1-r3` and `c4-r1` through `c4-r3`.
- Gateway routing aliases: `bf16` received 70 requests; `fp8` received 74.
  For this scenario those are gateway endpoint aliases, not precision claims:
  both upstream workers were launched with stock `dtype=auto` and
  `kv-cache-dtype=auto`.
- Three excluded preflight/warm-up requests are in `preflight/`.

## Files

- `c*-r*/canonical-requests.jsonl`: one normalized row per measured request;
  includes `gateway_routed_worker` from the gateway response header.
- `c*-r*/requests.jsonl`, `responses.jsonl`, and `dispatch-ledger.jsonl`:
  raw client, stream, and dispatch evidence.
- `c*-r*/telemetry.jsonl`: gateway-health sampling. The greedy gateway does not
  export per-worker Prometheus running/waiting counters, so those fields remain
  explicitly missing rather than inferred.
- `manifests/gateway.json`: gateway endpoint aliases, worker workflow IDs,
  stock configuration, and telemetry limitations.
- `gateway-state-before.json` / `gateway-state-after.json`: gateway routing
  state snapshots.

This result measures the two-stock-worker gateway configuration. It does not by
itself prove two GPUs are faster than one; compare the same frozen c4 traces to
a matched one-worker control before making that claim.
