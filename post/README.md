# Qwen stock pilot: results and interpretation

## Main finding

The profiling pipeline successfully sent five sequential requests to vLLM and collected consistent client and server measurements. This establishes that the basic measurement path works. It does not establish peak serving capacity, answer accuracy, or a benefit from changing KV-cache precision.

## Run context

- Source: `profiling/results/vast-reactome-qwen-stock-pilot-2026-11-14`.
- Model: `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`.
- Worker manifest reports vLLM 0.28.0; benchmark used AIPerf 0.13.0.
- Configuration: stock, KV-cache dtype recorded as `auto`, prefix caching enabled.
- Effective configuration verification: false. Do not label this a verified BF16 or FP8 KV run. AWQ describes weight quantization, not KV-cache precision.
- Requests: 5, concurrency: 1, streaming enabled, requested output cap: 256 tokens.
- Source inputs: supplied Reactome records; character-sized preview suite.
- Benchmark measurement duration: 41.543 seconds. Setup and export time are outside this measurement window.

## Client performance measurements

| Measurement | Result | Interpretation |
|---|---:|---|
| Completed requests | 5 of 5 | All measured requests completed |
| Reported request errors | 0% | No reported request failures |
| Mean time to first token (TTFT) | 3.216 s | Delay until the first output token |
| Median / p95 TTFT | 3.223 / 3.234 s | Five-sample estimates only |
| Mean request latency | 8.280 s | AIPerf client request-latency metric |
| Median / p95 request latency | 8.627 / 8.921 s | Five-sample estimates only |
| Request latency range | 7.244–8.926 s | Observed range, not a future guarantee |
| Mean client decode duration | 5.064 s | First-to-last-output interval |
| Mean request-level inter-token latency | 25.703 ms | Average of the five per-request ITL values |
| Total output throughput | 23.810 tokens/s | 990 output tokens divided by the benchmark window |
| Mean per-user decode throughput | 38.906 tokens/s | Generation-phase rate; different denominator from total throughput |
| Request throughput | 0.12025 requests/s | Observed throughput at concurrency 1, not maximum capacity |
| Input tokens | 58,792 total; 11,758.4 mean | Server-reported usage counts |
| Output tokens | 990 total; 198 mean | Actual generated tokens, not requested limits |

The approximately 38.91 tokens/s decode rate and 23.81 tokens/s overall output rate are compatible: the full benchmark includes time spent waiting for initial output. Do not use them interchangeably in comparisons.

TTFT accounts for approximately 39% of mean client request latency; subsequent decoding accounts for approximately 61%. TTFT includes more than pure GPU prefill time. The matching server aggregates below help explain it.

## Individual requests

Request indices follow the original request export. Session IDs end in `0000` through `0004` under namespace `reactome-v1-42`.

| Request | Input tokens | Output tokens | TTFT (s) | Request latency (s) | Mean ITL (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 11,700 | 223 | 3.196 | 8.899 | 25.690 |
| 2 | 11,708 | 177 | 3.194 | 7.705 | 25.626 |
| 3 | 11,805 | 211 | 3.223 | 8.627 | 25.735 |
| 4 | 11,847 | 157 | 3.234 | 7.244 | 25.709 |
| 5 | 11,732 | 222 | 3.235 | 8.926 | 25.753 |

TTFT and average ITL vary little across these five requests. Requests generating more output generally take longer, consistent with a similar decoding rate. This narrow input range cannot establish how performance scales with context length.

## Server metrics and cache behavior

The metrics endpoint was successfully collected; the export reports 135 fetches across the collection interval. This interval includes time outside the client benchmark window, so server counter rates need not exactly match client benchmark throughput.

| Measurement | Result | Interpretation |
|---|---:|---|
| Mean server queue time | 0.0328 ms | Negligible queue delay in this pilot |
| Maximum sampled waiting requests | 0 | No waiting queue observed at sample times |
| Maximum sampled running requests | 1 | Consistent with concurrency 1 |
| Mean server prefill time | 3.145 s | Most of the client TTFT is consistent with prompt processing |
| Mean server decode time | 5.101 s | Server timing boundaries differ from client first-to-last-output timing |
| Mean server end-to-end latency | 8.300 s | Close to client timing; not identical instrumentation |
| KV-cache utilization mean / maximum | 0.538% / 0.719% | Fraction of the configured KV-block pool, not total GPU-memory utilization |
| Preemptions | 0 | No recorded preemptions |
| Prefix-cache hits | 160 of 58,792 queried tokens | Approximately 0.272% token hit rate |
| Normal-stop completions | 5 | Server finish-reason counters report normal stops |
| Length / abort / error / repetition finishes | 0 each | No recorded truncation at the output cap or abnormal finishes |

The cache metric explicitly defines 1 as 100% utilization; the percentages above convert that fraction once. Low KV utilization does not mean total GPU memory use is low or that a particular higher concurrency is safe.

These observations show no queue or cache pressure under this light load. They do not locate the engine's saturation point. Cache conditions were not controlled: a small nonzero prefix-cache hit count means this is not a verified fully cold-cache run.

The committed server export provides aggregated statistics and histogram estimates. It does not provide a timestamped state trace joined to individual requests. Individual request queue times and cache states cannot be reconstructed from these aggregates. No GPU telemetry endpoint was successfully collected, so GPU utilization or memory-bandwidth bottlenecks cannot be concluded from these files.

## Data quality checks

- Five unique session records are present.
- Client request totals and server totals agree: 58,792 input tokens and 990 output tokens.
- The pilot-input and dataset-manifest hashes match the performance manifest.
- For each request, client TTFT plus client decode duration equals the exported client request latency.
- The run is marked complete, not cancelled, with an empty error summary.
- All outputs are shorter than the requested 256-token cap. AIPerf's output-length mismatch count of five reflects normal early stops here, not five failed requests.

## Important limitations

1. **This was approximately an 11.8K-input-token run, not an 8K run.** The suite's nominal target was 8,192, but it used character-based sizing. Use actual server counts when describing it. Regenerate with the matching tokenizer before claiming a fixed 8K workload.
2. **There are no accuracy scores in this result folder.** It contains performance metrics, not graded answers. Request success is not correctness. The committed request-level export contains metadata and metrics rather than generated answer text.
3. **Five requests are too few for stable tail estimates.** The reported p95 values are descriptive of this pilot only. They are not confidence intervals or service guarantees.
4. **Only concurrency 1 was tested.** There is no load curve, peak-throughput estimate, or measured saturation point.
5. **Only one stock configuration was tested.** There is no BF16/FP8/gateway comparison and no verified effective KV precision.
6. **Warmup, cache state and hardware/runtime provenance are incomplete.** The worker manifest is explicitly unverified. Record these before controlled comparisons.
7. **No SLO compliance conclusion is available.** Thresholds and a representative workload must be specified before evaluating whether latency is acceptable.
8. **This is insufficient training data for the routing regression model.** There is no diversity of load/precision conditions or per-request candidate-worker state needed for that purpose.

## What to preserve and do next

Preserve the original raw artifacts and treat this run as a successful measurement pilot. Next, finalize tokenizer-sized workloads, run the separate graded accuracy suite, and collect repeated performance runs at multiple concurrency levels. Use the same workload and decoding settings across configurations; verify effective precision and document the warmup/cache procedure. Save timestamped server state separately if later analysis needs request-to-state associations.

The useful eventual comparisons are throughput and latency versus concurrency, input length versus TTFT, output length versus decode duration, latency distributions, accuracy by configuration, and cache pressure under load. This pilot supports only preliminary individual-request observations and one measured load point.

## Original files to consult

- `performance/aiperf/profile_export_aiperf.json`: aggregate client measurements, units and benchmark configuration.
- `performance/aiperf/profile_export.jsonl`: five request-level metric records.
- `performance/aiperf/server_metrics_export.json`: server aggregate metrics and completion counters.
- `performance/pilot-input.jsonl`: exact five input prompts.
- `performance/manifest.json` and `performance/dataset-manifest.json`: run status, cache caveat and input provenance.
- `worker-manifest.json`: recorded serving configuration and its verification status.
- `run.json`: source provenance and references to artifacts.

Source commit reviewed: `84cb7052e1b75a1079632266478a7775054dbc19`. No new inference requests were made for this review.
