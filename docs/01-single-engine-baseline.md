# 1. Establish a single-engine baseline

## Goal

Learn the profiling workflow and establish a baseline for a single serving
engine. The Mac currently runs `llama3.1:8b` via Ollama; it does **not** run the
vendored vLLM/CUDA engine. Treat the Mac run as an API smoke test. Establish a
separate vLLM baseline on GB10 when available, using the same 27B checkpoint,
engine build and configuration intended for the later comparison.

## Local practice run

Follow [the installation instructions](../profiling/README.md), make sure
Ollama is running and lists `llama3.1:8b`, and run from the repository root:

```sh
sh profiling/local-smoke.sh
```

This sends one warmup and five measured streaming requests at concurrency 1.
Results are written to a timestamped `results/local-smoke-*` folder. Review
`profile_export_aiperf.csv`, `profile_export.jsonl`, `inputs.json` and the log.
Server token usage is used for metrics, but synthetic prompt length is
approximate because the script uses a different tokenizer. The output limit
is a cap; generation can stop early. No server metrics or GPU profiling is
enabled in this smoke test.

## Single vLLM worker on GB10

Follow sections 1–3 of the [GB10 runbook](GB10-RUNBOOK.md): confirm the exact
27B checkpoint is supported by the pinned vLLM build, launch the worker via the
venue's Fuzzball workflow, and verify streaming, token usage, effective dtype
and model identity. Run the short AIPerf example in that guide with the
matching tokenizer and verified endpoint. Begin with one small context and
gradually increase length and concurrency. Save the manifest and raw results;
record failures, actual output length and cache state.

Use the same stock engine and settings for the later BF16/FP8 tests. AIPerf's
client TTFT includes queueing and network delay; it is not direct GPU prefill
time. Collect server timing or GPU traces separately if available.

## Graph and compare

- Plot individual request TTFT and total latency, with units and sample count.
- Plot output tokens/s and completed requests/s at each tested concurrency.
- Plot input and output token counts so short or early-ended generations are
  visible; flag failed or timed-out requests.
- Compare repeated runs of the **same** worker and workload to understand
  variability. The Mac and GB10 use different hardware, model and runtime:
  show them as separate demonstrations, not a controlled speed comparison.

Five Mac requests are not enough for meaningful tail-latency claims. Reserve
the GB10 baseline's raw files for the matched comparisons in step 2.
