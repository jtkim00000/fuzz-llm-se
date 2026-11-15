# 1. Profile stock vLLM with Nemotron on one ThinkStation

## Purpose

Establish a reproducible, single-worker baseline using the unmodified vLLM
engine and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` on a GB10
ThinkStation. This checkpoint uses NVFP4 **weights**; its KV-cache dtype is a
separate setting. Record the effective stock KV dtype and all settings: stock
may resolve to BF16 or another format, so it is not automatically distinct
from the explicit precision experiments.

Nothing in this guide has been verified on the venue's actual hardware. The
earlier laptop smoke test used Ollama and Llama, not vLLM or this model.
The VAST-backed preparation and verified answer keys are described in
[profiling/README.md](../profiling/README.md).

## 1. Prepare and verify

1. Ask the Fuzzball operator to start one persistent GPU worker on one node and
   provide its OpenAI-compatible endpoint, metrics endpoint, image, model mount,
   and permitted authentication method. The second node should not run a
   competing serving worker during these single-engine measurements.
2. Choose and record an exact vLLM commit/image digest, driver/CUDA versions,
   model and tokenizer revisions, model load format, GPU identity, memory,
   max context, batching, prefix caching, speculative decoding, and sampling
   settings. Confirm the image actually runs the repo's vendored revision if
   claiming that it does. The [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
   provides a vLLM recipe and recommends a DGX Spark container; compatibility
   with our particular version and GB10 setup still requires a real test.
3. For **stock**, do not specify a KV-cache override; record the actual resolved
   dtype, scale source and attention backend from logs/instrumentation. The
   model is hybrid Mamba/attention; check which layers/state use KV cache and
   record Mamba state dtypes separately. Do not assume all state is BF16/FP8.
4. Verify `/v1/models`, a short streaming completion, server token usage,
   `/metrics`, context capacity and worker health. Increase context and
   concurrency gradually and record OOMs. Keep thinking mode and output
   semantics fixed across experiments; generated reasoning tokens count too.

## 2. Install AIPerf and run a pilot

Use the prepared cluster client or install AIPerf as described in
[profiling/README.md](../profiling/README.md). The example below uses a local
virtual environment; cluster runs use the environment path in that guide.
Set the following placeholders for the actual deployment. A local tokenizer
directory from the pinned model revision is preferable. Supply credentials by
the venue-approved method and never commit or print them.

```sh
WORKER_URL='http://WORKER_HOST:8000'
SERVED_MODEL='YOUR_SERVED_MODEL_NAME'
TOKENIZER='/path/to/pinned/tokenizer'
RUN_DIR='results/stock-8k-256-c1-rep1'

.venv-aiperf/bin/aiperf profile \
  --model "$SERVED_MODEL" --url "$WORKER_URL" \
  --endpoint-type chat --streaming --tokenizer "$TOKENIZER" \
  --use-server-token-count --concurrency 1 --request-count 5 \
  --warmup-request-count 1 --isl 8192 --osl 256 \
  --random-seed 42 --extra-inputs temperature:0 \
  --artifact-dir "$RUN_DIR" --ui none
```

This is a pilot, not an explicitly cold test: warmup may populate caches.
`--osl` is a target/cap, not a guarantee of generated length. Check actual
server usage and whether the server accepts AIPerf's output-limit parameter;
use `--use-legacy-max-tokens` only if the server needs it. Add
`--server-metrics http://METRICS_HOST:PORT/metrics` if the real metrics
endpoint is separate and accessible. AIPerf flags shown are for 0.13.0.

## 3. Collect the baseline matrix

Use fixed traces or saved seeded inputs and record their hash. Pilot context
sizes 8k and 65k, short and long output targets, and concurrency 1 and 2;
expand as capacity permits to context 8k/32k/65k/130k, output target
256/1024/4096 and concurrency 1/2/4. Context plus output must fit the model
limit; cache occupancy and hybrid state may constrain concurrency. Repeat
important cells at least three times. Save actual input/output lengths,
workload and cache state, errors and repetitions, not just requested values.

For cold runs use unique prefixes and verify cache hits are zero; warm the
model separately without contaminating measured prefixes. For warm sessions,
prefill a known prefix and replay follow-ups on the same worker, verifying
cache reuse via available server data. A synthetic single-turn pilot does not
measure warm sessions. Reset caches only with a verified venue-supported method.

## 4. Record and graph

Capture client TTFT, inter-token latency, end-to-end latency, output tokens/s,
request throughput, actual input/output/cached tokens, status and errors;
collect worker queue depth, active sequences, cache occupancy, prefill/decode
timings, utilization, memory, clocks and temperature where actually available.
Client TTFT includes queue/network, not pure prefill. Use server instruments
or GPU profiling for internal timing and bandwidth; mark unavailable values
null. AIPerf alone cannot observe every kernel or cache value.

Save each run's raw JSONL, summaries, trace hash, metadata, Git commit and
dirty state, image/model/tokenizer revisions, settings and warmup/cache method.
Keep secrets out. `results/` is ignored: archive results to team-approved
storage, such as VAST if provisioned. Plot TTFT and end-to-end latency versus
context, decode throughput versus context, output and concurrency, and
failures. Include sample counts and variation. This stock baseline is the
reference for [step 2](02-kv-precision-comparison.md).
