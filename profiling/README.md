# Profiling

For the Nemotron experiment on the venue's GB10 workstations, follow the
three sequential guides in [`docs/`](../docs/README.md). The older GB10
runbook is preserved as background, not as the current model plan.

## Local smoke test

The current Mac setup uses Ollama, not the vendored vLLM. Start Ollama and
ensure `ollama list` includes `llama3.1:8b`. Install the profiling client in an
isolated Python 3.11+ environment:

```sh
uv venv --python 3.11 .venv-aiperf
uv pip install --python .venv-aiperf/bin/python 'aiperf==0.13.0'
sh profiling/local-smoke.sh
```

The script sends one warmup and five measured streaming requests directly to
Ollama at localhost:11434, concurrency 1. It uses the built-in tokenizer for
synthetic prompts and server-reported token counts for metrics. Prompt lengths
are approximate because that tokenizer is not Llama's; the 64-token output
limit is a cap, not a guaranteed length. Results, inputs, logs and the base Git
commit are stored under `results/local-smoke-<UTC timestamp>/`. This commit
does not capture uncommitted changes. Results are ignored by Git.

## What the router needs

The execution guide specifies a performance table keyed by precision, context
tokens, output tokens and concurrency, with prefill time, decode tokens/s,
TTFT, repetitions and variability. AIPerf's per-request and aggregate exports
provide client-observed timing and token usage when the server supplies it;
TTFT includes queueing and network time, and is not pure prefill time. Use a
matching tokenizer and server instrumentation for actual token counts, cached
tokens, queue delay, prefill and decode timing. AIPerf collects compatible
Prometheus metrics only if those endpoints are exposed and accessible.

For GB10, pin the engine, model, kernel, driver and KV dtype; replay matched
cold and warm workloads for BF16 and FP8 at multiple lengths and loads, with
at least three repetitions. Save configurations, trace hashes, raw outputs and
errors. Record live queue and worker state at assignment for a load-aware
router. Split training and evaluation by session or trace, and reserve separate
final benchmark traces. Correctness and kernel-level GPU profiling require
additional tools and tests. Do not label the Mac results as BF16/FP8 KV-cache
measurements or use them as target-hardware training data.
