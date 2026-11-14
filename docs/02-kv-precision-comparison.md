# 2. Profile one BF16 worker and one FP8 worker separately

## Purpose and prerequisites

Use the exact Nemotron NVFP4 checkpoint and stock vLLM setup from
[step 1](01-single-engine-baseline.md). Run **one serving engine at a time**:
complete the BF16 series, then the FP8 series (or vice versa). These are two
single-worker configurations, not a dual-worker routing test. Use the same
ThinkStation when practical; otherwise measure node differences and document
them. Coordinate Fuzzball worker restarts and ensure no competing workload.

The repo's vLLM source includes `bfloat16` and `fp8` as KV dtype options;
confirm the deployed build supports them using `vllm serve --help`. The
typical overrides are `--kv-cache-dtype bfloat16` and `--kv-cache-dtype fp8`.
Add one override to the venue-approved launch from step 1, keep every other
setting fixed, then verify effective dtype, attention backend, FP8 scaling and
any skipped layers from logs or instrumentation. Hybrid Nemotron also has
Mamba state: record its dtype separately. Failure to verify means the cell is
unverified, not a valid BF16/FP8 comparison. Consult [vLLM KV-cache documentation](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/).

## Run the experiment

1. Start the BF16 worker; verify model output, effective configuration and
   available memory. Use the same AIPerf pilot command from step 1, changing
   `WORKER_URL` and using a distinct result directory labeled `bf16`.
2. Run the same fixed traces, cold/warm procedures, context/output lengths,
   request rates and concurrency levels as step 1. Repeat key cells at least
   three times. Save raw per-request results and actual token counts.
3. Stop/restart the worker using the approved Fuzzball procedure, then start
   the FP8 worker and verify its effective settings. Run the exact matched
   sequence with directories labeled `fp8`. Preserve seeds and trace hashes;
   record output-length differences and cache hits rather than assuming them.
4. Monitor worker health, queue/active work, GPU memory, utilization, clocks
   and temperature where available. Record prefill and decode timings from
   server instrumentation; TTFT is not pure prefill. Use independent GPU tools
   for bandwidth/kernel diagnosis if supported. Report missing values.
5. Run identical correctness tasks: executable code tests, known-answer
   repository questions and field-level JSON extraction. Report pass rates,
   sample sizes and flips; predeclare a tolerance before final comparisons.

Changing only KV dtype isolates the precision effect. If testing custom
kernels, run a **separate** stock-versus-custom comparison for each dtype,
with correctness tests and kernel/build versions recorded.

## Data for regression

For each measured request or session retain precision, trace/session/turn IDs,
context, newly processed and cached token counts when available, requested
output budget, actual generated tokens, concurrency, offered load, worker
queue/active state at assignment, TTFT, server prefill/decode timing when
available, end-to-end time, status and configuration manifest. Raw results,
manifests and trace hashes go in separate directories. Derived performance
tables should include units, sample count and variation; no such generator is
implemented yet. Low-load runs alone cannot teach load/queue behavior, so
collect loaded traces too.

## Graph and compare

- TTFT and full latency versus input/context size, separated by dtype, cold or
  warm state and concurrency; show replicate variation and failure counts.
- Per-request decode tokens/s versus context and actual output length; report
  aggregate throughput separately.
- KV-cache usage and hits, GPU memory and utilization where measured; show
  correctness pass rate per task and precision with sample sizes.
- Compare both series to step 1's **matched stock** baseline. If stock resolves
  to the same effective dtype and settings, describe it as the same effective
  configuration, not an independent quantization. Do not compare Mac numbers
  as if they came from the GB10.

Next: [train and benchmark the dual-worker gateway](03-gateway-system.md).
