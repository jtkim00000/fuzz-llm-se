# Profiling the 27B model on the GB10 workstations

> Historical background: the current team plan is NVIDIA Nemotron 3 Nano on
> vLLM. Follow [the updated three-step guides](README.md) for the experiment;
> this document retains earlier Qwen/SGLang assumptions and is not the current
> model-selection or run-order instruction.

This is a venue runbook, not a record of completed GB10 tests. Its goal is to
collect measurements for a session-routing cost model and then evaluate that
router. The Mac/Ollama smoke test is useful for learning AIPerf, but its
numbers do not describe these workers.

## 1. Agree on the actual stack before collecting data

The final proposal names `RadixArk/Qwen3.8-27B-NVFP4` with SGLang. This repo
contains vLLM instead. The RadixArk model card documents SGLang support, not a
validated vLLM/GB10 configuration. Confirm with the team which runtime and
checkpoint to use, and prove that exact combination starts and generates on
the actual GB10 before investing in a sweep. NVIDIA also publishes a separate
`nvidia/Qwen3.8-27B-NVFP4` checkpoint with a vLLM example; it is a different
artifact, so a change requires agreement and new baselines.

Use one copy of the same checkpoint per machine. NVFP4 weight precision is
separate from the BF16 or FP8 **KV-cache** precision being tested. Initially
run the stock engine; record custom-kernel experiments as separate versions.
For every configuration verify from startup logs and available instrumentation
which dtype and attention backend were actually selected. For a hybrid model,
confirm which layers and recurrent state the KV dtype controls; do not assume
all state is quantized. Consult the pinned engine's help and FP8 scaling
documentation: support, scale factors and flag behavior vary by version.

## 2. Prepare the workers and client

1. Have the Fuzzball owner launch two persistent workers on separate GB10
   machines, each exposing its own reachable OpenAI-compatible endpoint. Record
   the Fuzzball workflow IDs, internal/public URLs, and how to restart each
   worker. Do not assume a generic pool endpoint preserves worker identity.
2. Pin the vLLM source commit and container image digest, model and tokenizer
   revisions, custom kernel commit, CUDA/driver versions, and AIPerf version.
   Record GPU model, available memory, cache capacity, context limit, prefix
   caching, batching, sampling, speculative decoding and attention settings.
   Keep these identical between precision variants except for the intended
   change; record any exception. Use one model copy per node, without tensor
   parallelism across the two workers.
3. Select BF16 KV on worker A and FP8 KV on worker B using options confirmed
   against the actual engine (`vllm serve --help` and logs). In vLLM `auto`
   uses the model dtype; it is BF16 only if the effective model dtype is BF16.
   `--kv-cache-dtype fp8` requests FP8 but does not prove it is active. Verify
   maximum context, headroom and concurrency through progressively larger tests;
   leave room for output tokens and other model state. Record OOMs and limits.
4. Run AIPerf from a client able to reach both endpoints. Install it with the
   instructions in `profiling/README.md`; use a matching tokenizer revision,
   not `builtin`, for quantitative token-based results. Configure access
   credentials securely as directed by the venue; never save tokens in Git,
   command transcripts, or result metadata. If a proxy hides streaming, usage
   fields or metrics, use a permitted direct endpoint and record that route.
5. Check each endpoint's `/v1/models` and send a short streaming completion.
   Confirm the served model name, actual output, token usage, cache reporting
   and metrics endpoint. For vLLM, inspect `--enable-prompt-tokens-details`
   in the pinned version and enable it if supported. Use the model owner's
   tokenizer/chat template; validate counts against server usage.

Fuzzball allocates and manages the serving workflows; AIPerf issues HTTP
requests. Fuzzball does not execute each inference step as a separate job.
Do not invent cluster workflow commands: use the venue's actual configuration.

## 3. Start with one short cell

Replace the placeholders with the verified worker URL, **served model name**
and matching accessible tokenizer. Run from the repo root; AIPerf 0.13.0 is
pinned for this example. If authentication is needed, supply it through the
venue-approved environment/secret mechanism; do not include a token here.

```sh
WORKER_URL='http://WORKER_HOST:8000'
SERVED_MODEL='YOUR_SERVED_MODEL_NAME'
TOKENIZER='YOUR_MATCHING_TOKENIZER_PATH_OR_ID'
RUN_DIR='results/gb10-bf16-8k-256-c1-rep1'

.venv-aiperf/bin/aiperf profile \
  --model "$SERVED_MODEL" --url "$WORKER_URL" \
  --endpoint-type chat --streaming --tokenizer "$TOKENIZER" \
  --use-server-token-count --concurrency 1 --request-count 3 \
  --warmup-request-count 1 --isl 8192 --osl 256 \
  --random-seed 42 --extra-inputs temperature:0 \
  --artifact-dir "$RUN_DIR" --ui none
```

This is a **connectivity and measurement check**, not a controlled cold-cache
experiment: the warmup can populate the cache. `--osl` is a target/cap; early
EOS and server-specific behavior can change actual output length. Inspect
`inputs.json`, the per-request JSONL, summary CSV/JSON, logs, and API usage.
Verify measured input/output counts and that all requests succeeded. Check the
engine's supported output-limit parameter; `--use-legacy-max-tokens` may be
needed for a server that accepts `max_tokens` rather than
`max_completion_tokens`. AIPerf server metrics require a reachable Prometheus
endpoint; specify `--server-metrics URL` when the metrics port differs. If
metrics are unavailable, record that fact rather than interpreting missing
values as zero.

## 4. Collect data for the cost model

Use fixed, saved traces or fixed seeds and identical prompt content and arrival
schedule for both precision variants. Save tokenizer revision, trace file and
SHA-256, actual counts and all per-request records. Use temperature 0 and the
same sampling and output-limit policy. A fixed maximum does not guarantee equal
generated lengths; compare actual lengths and use a verified minimum/ignore-EOS
mechanism only if the server supports it. For realistic sessions preserve
correct history and session affinity; don't assume identical generated text.

The guide proposes this matrix; start with a small pilot and expand only as
capacity and time allow:

| Dimension | Values |
| --- | --- |
| Existing context | 8k, 32k, 65k, 130k tokens |
| New input | 1k, 8k, and full-context cold input |
| Output target | 256, 1,024, 4,096 tokens |
| Concurrency | 1, 2, 4 |
| Precision | BF16, FP8 |
| Repetitions | At least 3 per important cell |

Not every combination is feasible: total context plus output must fit the
configured limit, and available cache capacity limits concurrency. Treat the
matrix as a target, not a claim that all cells run. Prioritize 8k and 65k,
short and long output, concurrency 1 and 2, then expand. Run the same cells on
both workers. Repeat trials and report mean, standard deviation and failures.

**Cold:** create unique prefixes for different sessions and verify server
`cached_tokens` is zero or explain deviations. Reset/flush caches using a
verified engine endpoint or restart when necessary; use the same method for
both variants. Warm up the model separately without reusing the measured
prefixes. Resetting before a warmup does not guarantee the measured run is
cold. Never claim cold results unless cache status is checked.

**Warm:** prefill a known prefix, then send follow-up turns to the *same*
worker with the actual prefix/history preserved. Confirm cache-hit counts and
record cached versus newly processed tokens. Use real session traces and the
gateway's session ID contract; AIPerf's synthetic single-turn `--isl` command
above is not a warm-session test. Token counts and prior generated text can
diverge between runs, so record actuals and replay method.

For each request capture the information available at assignment: session
and turn ID, new/cached input tokens, output budget or a realistic output
estimate, worker ID, precision, timestamp, active/queued work and cache
occupancy if available. Save actual output tokens, TTFT, inter-token and total
latency, status/error, and server timing metrics. TTFT is a client-observed
proxy that includes queueing and network, not direct prefill time; record
server-side prefill/decode measurements when exposed. Decode token rate can be
estimated from streamed tokens and time after first token, with the exact
counting convention documented. GPU/kernel counters need separate supported
telemetry/profiling tools; AIPerf cannot invent unavailable measurements.

## 5. Load-aware routing and evaluation

Low-load data helps model service time. Also replay separate traces under low,
medium and near-saturation offered rates and capture worker state *before*
assignment. Compare request-rate/open-loop experiments with the concurrency
tests above: concurrency 1 alone cannot train queue behavior. Continuous
batching couples requests, so an isolated service time plus queue delay is an
approximation, not a guaranteed finish time. Measure loaded predictions and
avoid using actual output lengths or future queue state as online features.
Keep entire sessions/traces and whole runs separate across train/validation/
test splits, and reserve independent traces for final reporting. Record both
worker-specific measurements where feasible; a live request reveals only the
chosen worker's actual outcome, not its counterfactual latency.

Compare session-sticky BF16-only, FP8-only, mixed with a simple heuristic, and
mixed with the learned router using the same traces and rates. Warm up each
configuration equivalently, repeat headline runs, report p95 end-to-end
latency, TTFT, output tokens/s, completed requests/s, failures and variability.
Evaluate task correctness separately on both precisions and on the routed
system; predeclare an acceptable threshold. Do not claim a speedup or that FP8
is safe until measured.

## 6. Keep results reproducible

Use a distinct result directory per configuration, trace, load and repetition.
Alongside AIPerf's raw records, inputs, summaries and logs save a manifest with
run ID/time, Git commit **and dirty status/diff or clean revision**, image and
model/tokenizer revisions, endpoint identity (no secrets), dtype/scales,
context/batch/cache settings, hardware/software versions, trace hash, warmup
and cache procedure, seed, arrival rate/concurrency, requested versus actual
tokens, errors and any missing metrics. `results/` is ignored by Git: archive
it in team-approved storage. Derive a versioned `perf_table.json` from verified
raw records, including units, sample count and standard deviation; leave missing
fields null rather than guessing. The table generation, gateway telemetry and
Fuzzball worker definitions are still to be implemented in this repository.

## References

- [Proposal checkpoint and supported runtime](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
- [Separate NVIDIA checkpoint](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
- [Published GB10/SGLang measurements and conditions](https://github.com/pangoleen/qwen3.8-27b-dgx-spark-dflash2/blob/master/RESULTS.md)
- [vLLM quantized KV cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
- [AIPerf command-line options](https://docs.nvidia.com/aiperf/reference/command-line-options)
- [AIPerf trace replay](https://github.com/ai-dynamo/aiperf/blob/main/docs/benchmark-modes/trace-replay.md)
- [Fuzzball model-serving documentation](https://docs.ciq.com/fuzzball/advanced-features/serving-a-model)
