# Tomorrow-morning 30-minute BF16/FP8 campaign

This is a temporary, time-boxed replacement for the larger controlled
BF16/FP8 performance-characterization campaign. It changes **sample count
only**. Keep the same direct-node architecture, current collector, canonical
schema, VAST Reactome source, pinned tokenizer, matched engine settings, cache
procedure, and separate BF16/FP8 result folders.

Do not route through the gateway for this campaign. Run one direct profiling
job against each dedicated serving engine in parallel.

## Fixed workload matrix

| Dimension | Tomorrow-morning value |
| --- | --- |
| Formatted input-token targets | 512, 2,048, 8,192, 16,384 preferred; 512, 2,048, 4,096, 7,168 fallback |
| Controlled generated-token targets | 32 and 256 |
| Closed-loop client concurrency | 1 and 4 |
| Distinct prompts per length/output cell | 1 |
| Independent repetitions | 1 |
| Configurations | verified BF16 KV and verified FP8 KV |

`4 lengths × 2 outputs × 2 concurrencies × 1 prompt × 1 repetition = 16`
measured requests per precision, **32 total**. BF16 and FP8 run concurrently on
their separate nodes. This is deliberately not a 30-minute guarantee: stop and
preserve partial artifacts if a worker is unavailable or controlled 256-token
generation is materially slower than its verified request deadline.

## Block order and matching

Create two blocks per precision:

- one concurrency-1 block containing its eight length/output cells;
- one concurrency-4 block containing its eight length/output cells.

Shuffle each block with a recorded seed. Freeze prompt ID, VAST record selection,
formatted token count, output target, nonce, block position, and seed before
dispatch. Replay the exact same two blocks on BF16 and FP8. A loaded pair is a
matched workload, not an identical scheduler state.

There is one observation per cell, so report individual points and the overall
distribution only. Do not report cell p95/p99, repeat variability, or a
precision crossover from this compact run.

## Context and output controls

Before creating the trace, inspect live BF16 and FP8 launch settings and their
effective common `max_model_len`.

- Use the preferred 16,384 target only when **both** workers accept at least
  16,384 formatted input tokens plus 256 generated tokens under a verified
  common limit (normally 32,768).
- Otherwise use the fallback length set on **both** workers. Never truncate,
  downsize one precision only, or compare unequal length cells.

Build each VAST context from complete deterministic Reactome records, with no
repeated filler. Count the actual rendered chat-template tokens and record the
actual count/bucket. Keep thinking disabled and use the same instruction at all
lengths; ask for an extended structured catalogue rather than a short retrieval
answer.

Perform one bounded readiness check per worker to verify the actual serialized
generation request. If supported by the pinned vLLM endpoint, use the verified
combination of `min_tokens`, `max_tokens`, `ignore_eos`, and stop settings that
requests the 32 or 256 output target. Record the terminal usage and
`finish_reason`; `length` is expected under this synthetic timing condition.
If exact length control is unsupported, record the natural-length fallback and
actual output tokens. Do not claim fixed-length output in that case.

## Cache, warmup, and environment

Prefer prefix caching disabled on both dedicated benchmark workers only when the
serving owner authorizes and verifies that matching change. Otherwise retain it
on both, put a fixed-length unique nonce early in each prompt, and label the
condition `prefix-reuse-minimized, not verified cold`.

Use separate warmups with different prompt IDs/prefixes and short/long shapes;
exclude them from the 16 measured requests. Keep model revision, tokenizer,
sampling, GPU allocation, max-num-seqs, batch-token limit, KV allocation,
client location, and telemetry settings matched and recorded. Do not restart a
shared worker without serving-owner approval.

## Required records and export

For every measured request, retain the current canonical JSONL record and raw
artifacts: request/run IDs, logical prompt ID, formatted input tokens, requested
output cap, pre-dispatch running/waiting counts and sample age, TTFT, actual
output usage, E2E latency, dispatch/completion timestamps, status, finish
reason, error type, ledger record, raw stream events, and telemetry snapshots.

Also retain first/last visible generated-output timing when observed and report
the explicitly labelled average TPOT estimate only when output usage is greater
than one. Never manufacture per-token timings or use actual output length as an
online feature.

Create a compact table by precision, input bucket, output target, and
concurrency. Include actual input/output counts, TTFT, E2E, status, and the
paired FP8-minus-BF16 difference. Only report throughput at the whole-block
level for the concurrency-4 blocks.

## Scope and interpretation

This compact campaign is a morning feasibility/performance characterization, not
the eventual router training or quality-validation set. Preserve failures and
slow tails. The separate natural-output correctness check remains valuable, but
is outside this strict 30-minute performance budget unless the 20-task suite is
already known to fit in the remaining time. Do not interpret forced outputs as
answer-quality results.

Afterward, retain the direct-node outputs as a new active result directory; do
not overwrite the preserved `deprecated-direct-node-*` data.
