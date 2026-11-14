# 2. Compare BF16 and FP8 KV-cache workers

## Goal and setup

Measure the same 27B model with BF16 KV cache and FP8 KV cache on the GB10
machines. Run a persistent worker per machine via Fuzzball; benchmark each
directly, without the gateway, using the same engine build, checkpoint,
tokenizer, workload, sampling, cache policy and hardware allocation. Model
weight precision and KV-cache precision are different settings.

Before measuring, prove both workers load and generate, inspect logs to verify
effective dtype, scales and attention backend, and verify which layers or
model state the dtype affects. The proposed RadixArk checkpoint documents
SGLang; vLLM compatibility on the venue setup is not established. Do not
assume `auto` means BF16 unless the effective model dtype is BF16. See sections
1–2 of the [GB10 runbook](GB10-RUNBOOK.md).

## Run matched cells

1. Use the AIPerf command in section 3 of the runbook against each worker,
   starting with a short 8k-input, 256-output-target, concurrency-1 pilot.
   Use the exact served model name and matching tokenizer. Verify actual input,
   output and cached-token counts and that all requests succeeded.
2. Expand toward the runbook matrix: 8k, 32k, 65k and 130k context; 256,
   1,024 and 4,096 output targets; concurrency 1, 2 and 4; at least three
   repetitions for important cells. Confirm context plus output fits and log
   infeasible cells or OOMs. Prioritize a small subset first.
3. For cold runs use unique prefixes and a verified cache-reset procedure;
   warm up without contaminating the measured prefixes. Confirm cache counts.
   For warm runs, prefill a real prefix, replay follow-ups with session
   affinity and verify cache reuse. A single synthetic request cannot establish
   warm-session behavior.
4. Repeat the same fixed trace or seeded workload on both workers. Save trace
   hashes, config, raw records, server metrics, failures, actual lengths and
   variability. Record token-count conventions and server-side prefill/decode
   timings where available. TTFT is a proxy, not pure prefill.
5. Evaluate correctness independently using matched coding, repository QA and
   JSON tasks. Predeclare an acceptable threshold and record pass/fail by
   precision. Differences in output do not automatically mean errors.

## Graph and compare

- TTFT and end-to-end latency versus context length, BF16 and FP8 as separate
  series; include variation across repetitions.
- Decode tokens/s versus context length, separated by actual output length
  and concurrency. Show aggregate and per-request throughput separately.
- Cold and warm results side by side; include cache-hit counts and failures.
- Task pass rate by precision and category, with sample size.
- Compare the BF16 worker with the **GB10 vLLM baseline from step 1**, only if
  model, runtime, workload and settings match. Compare BF16 with FP8 within
  this step; do not compare either to the Mac as a precision effect.

Use the raw measurements to create a performance table with context/new
tokens, output tokens, concurrency, precision, timing, sample count and
variability. This table and its generator are not yet implemented. Avoid
filling missing cache or server-time fields with estimates presented as facts.
