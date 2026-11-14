# VAST Reactome Qwen stock pilot

This is the five-request, concurrency-one direct-vLLM pilot run on 2026-11-14.
The inputs are from the supplied VAST-mounted Reactome source
`/datasets/bio/reactome/ReactomePathways.txt` (SHA-256 recorded in `run.json`).

`run.json` is the entry point: it records provenance, all source-derived suite
inputs, the five executed sessions, and the paths to the raw artifacts. The
exported files needed for future training or analysis are:

- `performance/aiperf/profile_export.jsonl` — request-level AIPerf records.
- `performance/aiperf/profile_export_aiperf.json` — aggregate latency and
  throughput metrics.
- `performance/aiperf/server_metrics_export.json` — server metrics samples.
- `performance/pilot-input.jsonl` — the exact five prompts sent to vLLM.
- worker, run, command, and dataset manifests — reproducibility provenance.

This is an FP16/AWQ Qwen run with vLLM `kv_cache_dtype=auto`; it is not a
BF16-vs-FP8 KV-cache comparison and is too small to train a router.
