# Gateway benchmarking: slide-ready data

## Paste-ready slide text

### Scenario 1 — Least-Request Baseline Gateway

**Concurrency 1**

- **72 requests** completed successfully
- **2.196 s** mean time to first token (TTFT)
- **20.151 s** mean end-to-end latency
- **7.141 tokens/s** aggregate output throughput
- **7.108–7.174 tokens/s** throughput range across three repetitions
- **36 BF16 / 36 FP8** observed worker assignments

**Concurrency 4**

- **72 requests** completed successfully
- **2.580 s** mean time to first token (TTFT)
- **22.253 s** mean end-to-end latency
- **24.471 tokens/s** aggregate output throughput
- **23.681–25.251 tokens/s** throughput range across three repetitions
- **34 BF16 / 38 FP8** observed worker assignments

### Scenario 4 — Predictive Mixed-KV Gateway

**Concurrency 1**

- **72 requests** completed successfully
- **1.6% lower TTFT**: 2.160 s versus 2.196 s for the least-request baseline
- **1.9% lower end-to-end latency**: 19.766 s versus 20.151 s
- **2.0% higher aggregate output throughput**: 7.280 versus 7.141 tokens/s
- **7.251–7.308 tokens/s** throughput range across three repetitions
- The predictive policy routed **72 / 72 requests to FP8** under this low-load trace

**Concurrency 4**

- **72 requests** completed successfully
- **4.3% higher TTFT**: 2.692 s versus 2.580 s for the least-request baseline
- **3.0% higher end-to-end latency**: 22.919 s versus 22.253 s
- **2.7% lower aggregate output throughput**: 23.818 versus 24.471 tokens/s
- **23.670–23.927 tokens/s** throughput range across three repetitions
- The predictive policy actively used both engines: **39 BF16 / 33 FP8** assignments

### Predictive Gateway — Slide Conclusion

- **Predictive routing improved the low-concurrency case:** lower TTFT, lower end-to-end latency, and higher throughput on the same 72-request VAST trace.
- **The improvement is not yet universal:** the predictive policy regressed at concurrency 4, so it should be presented as a measured low-load improvement rather than an overall speedup claim.
- **Next step:** train or calibrate with concurrent gateway-path data and real pre-dispatch queue telemetry, then rerun the same held-out trace.

## Slide: Aggregate gateway benchmark results

**Title:** Predictive gateway: mixed BF16/FP8 KV-cache benchmark

**Subtitle:** Same frozen 144-request VAST suite; 72 requests at each concurrency, with three repetitions.

| Metric | Scenario 1: least-outstanding baseline, C=1 | Scenario 4: predictive mixed KV, C=1 | Scenario 1: least-outstanding baseline, C=4 | Scenario 4: predictive mixed KV, C=4 |
|---|---:|---:|---:|---:|
| Requests | 72 | 72 | 72 | 72 |
| Mean TTFT | 2.196 s | **2.160 s** | **2.580 s** | 2.692 s |
| Mean end-to-end latency | 20.151 s | **19.766 s** | **22.253 s** | 22.919 s |
| Aggregate output throughput | 7.141 tokens/s | **7.280 tokens/s** | **24.471 tokens/s** | 23.818 tokens/s |
| Throughput range across repetitions | 7.108–7.174 | 7.251–7.308 | 23.681–25.251 | 23.670–23.927 |
| BF16 / FP8 routed requests | 36 / 36 | 0 / 72 | 34 / 38 | 39 / 33 |

**Source data:**

- Baseline: `benchmarking/gateway-scenario1-stock-2026-11-15/`
- Predictive: `benchmarking/gateway-scenario4-predictive-bf16-fp8-2026-11-15/`

## Slide: What the predictive gateway improved

**Title:** Predictive routing improved the low-concurrency path

- At **concurrency 1**, predictive routing reduced mean TTFT by **1.6%** (2.196 s → 2.160 s).
- It reduced mean end-to-end latency by **1.9%** (20.151 s → 19.766 s).
- It increased aggregate output throughput by **2.0%** (7.141 → 7.280 tokens/s).
- Under this low-load trace, the learned policy chose FP8 for all 72 requests.

## Slide: Honest interpretation and next step

**Title:** Improvement is load-dependent, not yet a universal win

- At **concurrency 4**, the predictive run was slower: TTFT **+4.3%**, mean E2E latency **+3.0%**, and output throughput **−2.7%** versus the least-outstanding baseline.
- The C=4 predictive split used both engines (39 BF16 / 33 FP8), confirming active mixed-worker routing rather than a failed worker fallback.
- Therefore, the defensible claim is: **the predictive gateway improved the measured low-concurrency case, but this benchmark does not establish an overall improvement under load.**
- Next experiment: train or calibrate on concurrent gateway-path data with real pre-dispatch queue/worker telemetry, then replay the same held-out arrival trace against both policies.

## Speaker-note methodology

Scenario 1 was a two-worker stock-vLLM, least-outstanding-requests baseline.
Scenario 4 used one verified BF16-KV worker and one verified FP8-E4M3FN-KV
worker behind the learned minimum-predicted-completion-time policy. Both used
the same served model, request suite, output limits, concurrency levels, and
three repetitions. The worker configurations differ between scenarios, so this
is a policy-plus-mixed-KV comparison, not an isolated router-only ablation.

All 288 benchmark requests succeeded. Gateway response headers recorded the
actual selected worker. The collection path did not provide per-worker
running/waiting request counters; these are left missing in canonical records
rather than inferred.
