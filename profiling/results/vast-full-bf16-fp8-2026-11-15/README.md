# Full matched VAST BF16/FP8 direct-node campaign

This is the complete controlled performance-characterization campaign, separate
from the earlier 32-request compact pilot and the older 100-prompt run.

## Coverage

Each configuration has 144 successful measured requests (288 total): four
formatted-input targets (fallback 512, 2,048, 4,096, and 7,168 because both
workers had an 8,192-token context limit), two forced output targets (32 and
256), closed-loop concurrency 1 and 4, three deterministic prompt examples per
cell, and three independent randomized repetitions. Each `c{1|4}-r{1|2|3}`
directory has one 24-request block. Traces/prompts/order are identical across
BF16 and FP8 for a given condition/repetition.

`bf16/` is verified bfloat16 attention KV with FLASH_ATTN; `fp8/` is verified
FP8 attention KV with FLASHINFER. Both use the same Qwen3.8 checkpoint,
vLLM version, configured batch limits, and source/tokenizer revision. Prefix
caching was enabled; the condition is `prefix-reuse-minimized, not verified
cold`, not a cold-cache claim.

## Analysis inputs

`ENGINE/c*-r*/canonical-requests.jsonl` contains the graph/training-table
records: request IDs, worker/condition, local pre-dispatch input tokens,
requested output cap, running/waiting snapshots, TTFT, actual output tokens,
end-to-end latency, dispatch/completion timestamps, status, and finish reason.
`dispatch-ledger.jsonl`, `responses.jsonl`, and `telemetry.jsonl` retain raw
timing, streaming, and 200-ms telemetry evidence. Configuration/provenance,
worker manifests, frozen traces/hashes, and preflight/warmup evidence are also
preserved for every engine.

Use only pre-dispatch fields as routing-model features. TTFT, final latency,
actual outputs, and later telemetry are labels/diagnostics. Keep repetitions and
whole logical prompts grouped when splitting train/test data. Compare exact
condition/repetition pairs; do not claim p95/p99 from three prompt examples per
cell, and do not infer quality parity from these synthetic forced-output tests.
