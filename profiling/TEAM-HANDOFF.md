# Serving-engine profiling: context and deployment handoff

This document can be given to another LLM chat or teammate configuring the model
server. Its purpose is to explain the experiment and collect verified deployment
information so the profiling owner can run the existing benchmark client.

## Project and division of responsibilities

We are evaluating a precision-aware gateway in front of vLLM workers managed by
Fuzzball. The current target is
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` on NVIDIA GB10 machines.
NVFP4 is the checkpoint's weight format. BF16/FP8 are separate attention KV-cache
settings. This hybrid model also has Mamba/recurrent state; its dtype and reuse
must be recorded separately.

The serving team launches the model, verifies its effective configuration, exposes
reachable endpoints and metrics, and coordinates worker restarts/cache procedures.
The profiling team owns fixed inputs, AIPerf traffic generation, objective accuracy
grading, raw results, configuration records, and comparisons.

Fuzzball manages worker/job placement and lifetime. vLLM schedules inference and
manages its cache. AIPerf sends HTTP requests to vLLM's serving port. Open WebUI is
another client of that API and is not needed for benchmark execution.

```text
Shared VAST source records -> prepared/versioned input suite
                                      |
                     CPU benchmark job (AIPerf + accuracy runner)
                                      |
                     HTTP -> vLLM API -> loaded model on GB10
                                      |
                     profiling-results volume: responses, metrics, scores
```

Any additional frontend, LocalAI service, proxy, or gateway between the client and
vLLM must be identified. To claim a direct stock-vLLM baseline, provide the worker's
direct permitted endpoint; record any intervening proxy and its streaming behavior.

## Load-aware extension now in the repository

See TRAINING.md for the added collection/trace/campaign/export/training scripts.
The benchmark router can capture all candidate states, maintain session affinity,
and compare simple versus learned policies. This does not deploy the production
gateway. Exact token-level percentiles and unavailable server/hardware telemetry
remain missing until supported instrumentation is configured. Serving teammates
must provide actual metric names/labels and exporter endpoints, latency targets,
quality thresholds, cache procedures, and exact launch arguments in addition to
the original handoff below. The extended tools are now deployed at `/results/baseline-prep-qzFEdMaQ/tools`;
the existing AIPerf environment now includes scikit-learn 1.7.2. The baseline
paths below predate this extension; use TRAINING.md for the current entry points.

## Current preparation status

As established during setup on November 14, 2026:

- Fuzzball CLI and agent version: 4.3.1; cluster: `fb-hack`.
- Shared datasets volume: `datasets`, provisioner `shared`, normally mounted at
  `/datasets` for preparation. We only read its source files.
- Writable output volume: `profiling-results`, provisioner `team2`, approximately
  18.6 GiB, mounted at `/results` in the benchmark jobs.
- Source selected: `/datasets/bio/reactome/ReactomePathways.txt`.
- Prepared scripts and preview inputs: `/results/baseline-prep-IQ1zkCdX/`.
- Installed AIPerf 0.13.0 environment: `/results/benchmark-env-xzAhY13Y/venv/`.
  It is an arm64 environment, verified in the compiler-equipped `python:3.11`
  container. Preserve compatible architecture/runtime or rebuild the environment.
- Dataset preparation workflow: `a999ea50-a862-4d2d-9925-0844b67f10ad` — finished.
- Successful client install workflow: `0b4f2413-901f-40e5-9e5a-15b1f230b91a` — finished.
  The earlier install `35231fda-9be9-4cb9-b56c-4e1b0b9949ec` failed because the slim
  image lacked a compiler; its partial environment must not be used.
- Source data was parsed and all 20 answer keys checked against their prompt
  contexts. Offline tests verify strict scoring and failure accounting.
- Prepared input lengths are still character-based previews. They must be rebuilt
  using the pinned tokenizer before token-sized baseline claims.
- No actual Nemotron serving performance or model accuracy result has been
  established by these preparation jobs. Confirm current deployment state anew.

These are paths inside mounted cluster jobs, not files on the Mac. Keep model
weights/download caches on the serving team's separate writable model volume.

## The experiment

First characterize one worker at a time:

1. Stock vLLM: no KV override; record its effective KV dtype.
2. Explicit BF16 KV.
3. Explicit FP8 KV, only if supported and verified on this exact stack.

Keep the checkpoint, image/build, hardware allocation, context/batching settings,
prefix caching, thinking mode, sampling, and other settings fixed except the
intended precision change. If stock resolves to the same configuration as an
explicit variant, report it as a duplicate effective configuration.

Later evaluate homogeneous two-worker baselines and mixed BF16/FP8 gateway
policies using the same total hardware, held-out traces and session affinity.
The current starter suite does not implement that final gateway experiment.

Initial single-worker performance matrix:

| Input | Output cap | Concurrent requests |
| --- | ---: | ---: |
| Approximately 8K tokens | 256 | 1 |
| Same context set, longer instructions | 1,024 | 1 |
| Identical short-answer inputs | 256 | 2 |

Use 100 inputs per performance cell and at least 3 repetitions for important
comparisons. Counts/duration may need increasing for stable tail estimates.
Output caps are not guaranteed lengths. Preserve actual counts and reasoning-token
semantics. These runs use concurrency control, not a fixed offered-rate schedule.
Larger contexts, offered-rate sweeps and real multi-turn/cache tests are later work.

Accuracy is run separately, serially: 20 custom retrieval/JSON extraction tasks
from supplied Reactome records, with 10 single-record and 10 three-record tasks.
Answer-bearing records are near beginning/middle/end by record count. The source
records appear inside each prompt; mounting VAST does not give the model automatic
file access. Answer keys are separate and never sent to the model.

The grader requires the requested JSON shape, exact string field values and array
order. Wrong answers, invalid/duplicate-key JSON, extra prose, and truncated
responses fail. Request errors remain in the denominator. This is a pilot for
context use and format correctness, not a coding benchmark or biomedical knowledge
score. EvalPlus/RULER were discussed but are not installed evaluation suites here.

## What the serving team must provide

Please return confirmed values with their evidence source. Distinguish requested
launch settings from effective settings observed after startup. Unknown values
remain `null` and are described in `missing_measurements`/`notes`.

### Connection and storage handoff

These values configure `profiling/workflows/run-baseline.template.yaml`:

| Workflow variable | Required information |
| --- | --- |
| `WORKER_URL` | Base HTTP(S) address of the individual vLLM worker, e.g. `http://HOST:8000`, without `/v1`. Must be reachable from the cluster CPU benchmark job, not just from the Mac/browser. |
| `SERVED_MODEL` | Exact model ID returned by that endpoint's `/v1/models`; an alias may differ from the checkpoint name. |
| `METRICS_URL` | Reachable Prometheus endpoint, often `http://HOST:8000/metrics`. If unavailable, state this; remove the template guard/flag appropriately and record missing telemetry. |
| `TOKENIZER` | Local path to the matching pinned tokenizer directory inside the benchmark container, plus the volume name and mount configuration that make it accessible. |
| `WORKER_MANIFEST` | Path inside the benchmark job to the filled worker JSON, saved on writable storage. |
| `SUITE` | Profiling owner sets this to the finalized input directory after tokenization. Planned path: `/results/inputs/reactome-8k-v1`. |
| `CLIENT_ENV` | Already set to `/results/benchmark-env-xzAhY13Y`; change only if rebuilding/moving the client. |
| `TOOLS` | Already set to `/results/baseline-prep-IQ1zkCdX/tools`; update if deploying a newly generated tool bundle. |

Also provide the permitted authentication mechanism, secret reference and access
scope, any proxy/tunnel requirements, and who can restart the worker. Do not paste
credentials into chat, JSON manifests, source files, command transcripts or logs.
A Mac-local port-forward is not automatically reachable from a cluster job.
The tokenizer path and its mount are required even though model weights do not
need to be mounted into the benchmark client.

The simple accuracy client accepts `BENCHMARK_API_KEY` via approved environment
injection. Authenticated AIPerf integration still needs the venue-approved method;
the wrapper currently refuses to run performance mode with that variable set,
rather than placing the key into a saved command. Identify this requirement early.

### Worker manifest fields

Fill a COPY of `profiling/worker-manifest.template.json`; keep separate manifests
for different workers/precision configurations. Make the filled copy available on
`profiling-results` and reference its container path in `WORKER_MANIFEST`.

| JSON field | Required value / evidence |
| --- | --- |
| `effective_configuration_verified` | Remains `false` until relevant effective settings have been checked; disclose any unobservable fields. |
| `fuzzball_workflow_id` | The LLM-serving workflow ID, not the profiling/preparation job. |
| `worker_id` | Stable worker/node/replica identity; include host information in notes if useful. |
| `launch_arguments` | Exact effective launch arguments with secrets removed, including relevant environment overrides. |
| `engine_version` | Running vLLM version. |
| `engine_commit` | Exact source commit used in the image. Vendored source in the repository does not prove the container uses it. |
| `image_digest` | Running serving image digest, with its image reference in notes. A tag alone is insufficient provenance. |
| `model_checkpoint` | Confirm the planned NVIDIA Nemotron checkpoint or explicitly report a change. |
| `model_revision` | Exact downloaded model revision/snapshot commit. |
| `tokenizer_revision` | Matching tokenizer revision; record template overrides if used. |
| `served_model_id` | Exact API model ID; must agree with `SERVED_MODEL`. |
| `effective_kv_dtype` | Actual resolved KV-cache dtype, from startup logs/instrumentation. Do not infer it from weights, `auto`, or workflow name. |
| `kv_scales` | FP8 scaling method/source, supported details, and skipped layers if applicable; otherwise explain non-applicability. |
| `attention_backend` | Actual selected attention backend and any fallback. |
| `mamba_state_dtypes` | Effective recurrent-state dtypes, separately from attention KV dtype; state when not observable. |
| `max_model_len` | Effective integer context limit; input/history plus output must fit. |
| `batching_settings` | Object with effective limits, e.g. `max_num_seqs`, `max_num_batched_tokens`, chunked-prefill settings. |
| `prefix_caching` | Confirmed boolean; record hybrid-state/prefix reuse limitations in notes. |
| `thinking_mode` | Exact model/template configuration controlling thinking/reasoning. Keep it fixed across comparisons. |
| `speculative_decoding` | `false` if verified disabled, otherwise its effective configuration object. |
| `gpu` | Confirm NVIDIA GB10. |
| `gpu_count` | Count allocated to this worker; verify the template's `1`. |
| `system_memory_gib` | Total unified system memory in GiB; do not label it dedicated GPU VRAM. Add measured usable memory in notes when available. |
| `driver_version` | Host NVIDIA driver version. |
| `cuda_version` | Runtime version actually used by the serving container; distinguish it from driver-reported CUDA compatibility. |
| `warmup_and_cache_procedure` | Agreed execution warmup, cache reset/restart procedure, and evidence used to verify cold/warm state. |
| `missing_measurements` | List of unavailable settings/metrics and reasons. Missing is not zero. |
| `notes` | Provenance/evidence locations, exceptions, topology, template overrides and other material settings. |

Evidence can include redacted launch configuration, startup log excerpts, image
metadata and node diagnostics. Provide the exact launch flags in addition to
resolved values. Do not claim that a requested BF16/FP8 option proves support.

### Measurement readiness

Confirm these or report their absence:

- `/v1/models`, a short completion, streaming, and token usage work.
- Whether output limits use `max_tokens` or `max_completion_tokens`. The runner
  exposes `--output-limit-field`; its current default is `max_tokens`.
- What counts as generated output, including separate reasoning content/tokens.
- Metrics for queue depth, active work, cache occupancy/hits/preemptions and
  server prefill/decode timing, where exposed. We do not assume all are available.
- Worker identity remains stable; a generic load-balanced pool is not adequate
  for identifying a single-worker baseline.
- A quiet benchmark window without Open WebUI traffic/competing jobs, with a
  restart contact and a supported way to establish cache conditions.

TTFT includes queueing, network and serving overhead; it is not pure prefill.
AIPerf cannot derive unavailable GPU bandwidth, kernel, cached-token or server
phase measurements. Missing observations remain missing.

## What happens once the handoff is complete

1. Add the tokenizer mount to the benchmark workflow; transfer/fill the manifest.
2. Profiling owner rebuilds finalized input traces with the pinned local tokenizer,
   verifies answer keys and hashes, and checks real server input counts/context fit.
3. Fill the workflow variables and validate its YAML through Fuzzball.
4. Submit the pilot: model discovery, a completion check, 5 streaming performance
   requests, then 3 separately graded accuracy tasks.
5. Inspect responses, token counts, streaming timing, metrics, failures and grader
   outputs. Coordinate cache reset/re-warm if the pilot populated measured inputs.
6. Run the full performance matrix/repetitions and all 20 accuracy tasks. Repeat
   on matched explicit KV configurations after the stock baseline.

The pilot workflow is `profiling/workflows/run-baseline.template.yaml`. From the
repository root, after filling its deployment placeholders:

```sh
fuzzball workflow validate profiling/workflows/run-baseline.template.yaml
fuzzball workflow start profiling/workflows/run-baseline.template.yaml --name stock-baseline-pilot
# Use the returned workflow ID:
fuzzball workflow log WORKFLOW_ID baseline-pilot
```

Every run gets a unique directory on `profiling-results` with the filled worker
manifest, dataset manifest/hashes, AIPerf exports and console log, raw accuracy
responses and accuracy summaries. The worker manifest records deployment facts;
the separate run manifest records execution. No score/speedup should be reported
until the actual inference run succeeds and its effective configuration is verified.

## Repository and deployed tool versions

`profiling/README.md` contains operational setup details. `docs/01-...`, `02-...`,
and `03-...` describe the stock, KV-precision, and gateway stages respectively.
`docs/GB10-RUNBOOK.md` is historical Qwen/SGLang background, superseded by the
numbered Nemotron/vLLM guides. Earlier Mac/Ollama smoke measurements are not GB10
training data or precision baselines.

Editing repository Python files does not update already-deployed copies under
`/results/.../tools`. To deploy current tools, use
`python3 profiling/make_prepare_workflow.py`, validate/submit the generated job,
and record the newly printed root path. Do not overwrite versioned inputs/results.
The current tool builds preview inputs by default; matching-tokenizer finalization
remains a separate explicit step.

## Suggested reply from the serving team's chat

Please return:

1. Connection table: base API URL, served model ID, metrics URL, auth mechanism
   (no secret value), benchmark-to-worker route.
2. Tokenizer handoff: exact revision, volume name, container mount/path.
3. Filled worker-manifest JSON and redacted evidence for effective configuration.
4. Quiet benchmark window, cache/warmup procedure, restart contact.
5. Remaining unknowns or unsupported features. Do not invent values to fill fields.

The profiling owner will perform the input finalization, pilot and evaluation.
