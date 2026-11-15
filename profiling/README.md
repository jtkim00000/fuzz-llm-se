# VAST-backed profiling and accuracy baseline

This directory replaces the old Mac/Ollama smoke script, local workload previews,
and run-folder scaffolding. Measured results live on the `profiling-results`
volume; no local `results/` directory is needed for the cluster workflow.

For teammates configuring the serving worker or another LLM chat, see
[the deployment handoff and experiment context](TEAM-HANDOFF.md).

## Load-aware collection and regression training

The full sequence is documented in [TRAINING.md](TRAINING.md): configurable load
sweeps, fixed arrival schedules, multi-turn sessions, candidate-worker snapshots,
sticky benchmark routing, detailed accuracy diagnostics, training-table export,
and grouped-split forest training/evaluation. Use `collection-config.template.json`
and `campaign.template.json`, then `workflows/run-training.template.yaml`.

The extended tools are deployed at `/results/baseline-prep-qzFEdMaQ/tools`, and the existing
client `/results/benchmark-env-xzAhY13Y/venv` now includes scikit-learn 1.7.2.
The earlier baseline preparation paths below remain historical references.
Future repository edits still require regenerating/redeploying the tool bundle. No serving requests are launched by editing
these templates. Matching tokenizer and actual endpoint/configuration values are
still required. See TRAINING.md for exact measurement limitations and missing data.

## Start here

1. Reuse the existing prepared suite and AIPerf environment listed below.
2. Obtain the worker URL/model ID, metrics URL, pinned tokenizer mount/revision,
   and actual worker configuration from the serving team.
3. Fill a copy of `worker-manifest.template.json` on the writable results volume.
4. Generate a finalized token-sized suite using the instructions below.
5. Fill `workflows/run-baseline.template.yaml`, validate it, then submit it. It
   runs endpoint preflight, 5 streaming performance requests, and 3 accuracy tasks.
6. Inspect raw outputs and token counts before removing pilot limits, increasing
   repetitions, or comparing precision settings. Coordinate cache resets/warmup.

The pilot template includes a metrics URL placeholder; set it to the actual
Prometheus endpoint. Authentication, if required, must use the approved venue
secret mechanism; protected AIPerf endpoints need that integration before use.
The template is not submitted automatically and will refuse unresolved values.

## Repository commands

From the repository root, run the offline tests (no model, GPU, or network):

```sh
python3 -m unittest discover -s profiling/tests -v
```

To regenerate the preparation job from the current tracked tools:

```sh
python3 profiling/make_prepare_workflow.py
fuzzball workflow validate profiling/generated/prepare-baseline.yaml
fuzzball workflow start profiling/generated/prepare-baseline.yaml --name profiling-prepare-baseline
```

`profiling/generated/` is ignored; regenerate whenever the source scripts change.
Preparation reads the original datasets and writes a NEW preview directory; it
never modifies or overwrites the existing suite. The generated job embeds the
current source, avoiding dependence on a public Git checkout inside the cluster.

A fresh client install is available if the existing environment is unavailable:

```sh
fuzzball workflow validate profiling/workflows/install-client.yaml
fuzzball workflow start profiling/workflows/install-client.yaml --name profiling-install-client
```

This installs AIPerf 0.13.0 in a unique persistent environment and saves a resolved
dependency lock file and CLI help. The full Python image includes the compiler
needed by the ARM `crick` dependency. Client setup can take several minutes.
This requires outbound package access. Both workflow images are tags; record/pin
actual image digests before final reproducibility claims.


## What is already on the cluster

Persistent mount paths refer to the profiling-results volume, not paths on the Mac.

Prepared tools and input suite:
`/results/baseline-prep-IQ1zkCdX/`

- `tools/build_suite.py`: reads the provided Reactome TSV and builds traces/keys.
- `tools/grade.py`: strict JSON + exact record-value scorer; never executes outputs.
- `tools/selftest.py`: validates all answer keys against the supplied prompt context.
- `tools/run_baseline.py`: preflight, performance, and accuracy entry points.
- `tools/worker-manifest.template.json`: effective configuration to fill with the team.
- `inputs/reactome-v1-preview/`: 100 short-answer prompts, 100 longer-answer variants
  sharing their contexts, 20 separate accuracy prompts and keys, hashes and labels.

Preparation workflow: a999ea50-a862-4d2d-9925-0844b67f10ad.
Source: `/datasets/bio/reactome/ReactomePathways.txt`.
Source SHA-256: b8103a50d625ffb6acd18cddffcce0ceb0fdf36b19392d104c32bb1fcc7e2f9f.
Attribution/source and license references are saved with inputs.
The source volume was only read by the preparation script. The workflow uses the
existing simple mount syntax; a mount-level read-only restriction is not asserted.

## What this evaluates

Custom context retrieval and JSON extraction from real supplied records; this is
not biomedical expertise, code correctness, RULER, or an official benchmark score.
20 pilot tasks: 10 single-record and 10 three-record extraction tasks. Relevant
records occur near beginning/middle/end by record count. All exact keys/values,
array order, and JSON validity are required by the prompt. Duplicate JSON keys,
Markdown fences, extra fields, extra prose, wrong values, and truncated responses
fail. Errors stay in the accuracy denominator. Report categories and counts; a
20-item pilot cannot establish negligible quality degradation.

The preview uses a character budget, NOT a matching tokenizer. Input token counts
are null. Explicit --allow-preview permits smoke tests only. No 8K-token baseline
claim is valid until rebuilding with the pinned tokenizer and checking server usage.
The long answer output cap is 1,024, short 256. Accuracy cap is 2,048 including any
reasoning tokens charged by the server. Verify thinking mode and output semantics;
if changing caps, create a new suite and repeat every configuration.

## Remaining deployment handoff

Need the worker URL reachable FROM the CPU job, served model ID, permitted auth,
metrics URL, pinned local tokenizer files accessible inside the job, actual KV
precision/backend/scales/Mamba dtypes, context/batch settings, thinking mode,
image/model revisions and a quiet benchmark window. A service exposed only through
a Mac-local tunnel is not automatically reachable from the cluster job.

## Finalize inputs once tokenizer is mounted

Run INSIDE a Python environment with transformers installed, using an agreed
mount for the pinned tokenizer. The builder loads locally only with remote code
disabled. Never overwrite the preview or a finalized suite.

    python /results/baseline-prep-IQ1zkCdX/tools/build_suite.py \
      --source /datasets/bio/reactome/ReactomePathways.txt \
      --out /results/inputs/reactome-8k-v1 \
      --tokenizer /ACTUAL/PINNED/TOKENIZER --tokens 8192

    python /results/baseline-prep-IQ1zkCdX/tools/selftest.py \
      /results/inputs/reactome-8k-v1

Targets allow slack for instructions but exclude chat-template overhead. Verify
actual token counts and context capacity. Record tokenizer revision separately;
the path alone is not proof of a pinned revision. Freeze/hashes identify inputs.

## Launching tests

Use `workflows/run-baseline.template.yaml` after filling in real paths and endpoint values.
It runs preflight, a 5-request streaming pilot, and 3 accuracy tasks initially. The runner never
starts/restarts workers, clears caches, or claims effective precision from labels.
Each invocation requires a worker manifest; unknowns stay null and unverified.
The client environment is architecture-specific (currently arm64). Reuse it in
python:3.11 on the same architecture, or build a pinned client image. For final
measurements pin its image digest and save the actual client image/node identity.

Initial AIPerf installation workflow (failed: compiler absent): 35231fda-9be9-4cb9-b56c-4e1b0b9949ec.
Compiler-enabled retry: 0b4f2413-901f-40e5-9e5a-15b1f230b91a.
The retry environment is `/results/benchmark-env-xzAhY13Y`. Installation and the AIPerf help check completed successfully; do not use the failed environment `/results/benchmark-env-o8574gwP`.

After the pilot succeeds, use the performance command with --limit omitted to
run all 100 records. First run short concurrency 1, then long concurrency 1,
then short concurrency 2. Use unique result folders and at least 3 important
replicates. Warm model execution using unrelated prefixes. Reset the cache only
with the worker owner's verified method between matched repetitions/configurations;
otherwise unchanged traces may reuse earlier cache entries. Record the method and
measured cache hits. No cache or pure prefill claim comes from client TTFT alone.

Accuracy mode is a serial non-streaming client and provides scores/raw responses,
not the official performance timing measurement. Run separately from load tests.
It sends ONLY accuracy-prompts.jsonl text, not answer keys. No guided-JSON mode
is enabled; that would be a separate explicitly controlled experiment.

For a pilot add --limit 3, then remove it for all 20 tasks:

    "$CLIENT_ENV/venv/bin/python" "$TOOLS/run_baseline.py" accuracy \
      --suite "$SUITE" --out "$RUN_DIR/accuracy" \
      --url "$WORKER_URL" --model "$SERVED_MODEL" --configuration stock \
      --worker-manifest "$WORKER_MANIFEST"

Simple accuracy authentication uses BENCHMARK_API_KEY from the approved secret
injection mechanism. Do not store keys in JSON, source, or logs. Authenticated
AIPerf performance needs the venue-approved integration; the runner currently
fails explicitly if BENCHMARK_API_KEY is supplied for performance, rather than
putting it into a logged CLI. Use a permitted unauthenticated internal endpoint
or finish that integration before a protected-endpoint run. URL must be a base
URL without /v1, query credentials or embedded credentials.

Raw AIPerf files, console output, response JSONL, accuracy summaries and manifests
are retained per run. Accuracy summaries include failures, truncation and category/
position counts. Strict matching ignores object key order but preserves array order.
AIPerf/client timings are not server-only prefill or decode timings.

## Verification completed

Remote: real source TSV parsed; 20 correct/incorrect answer controls checked.
Local: strict grader edge cases and end-to-end mocked correct/wrong/transport-error/
truncated-response scoring. No actual vLLM result or precision speedup exists yet.
