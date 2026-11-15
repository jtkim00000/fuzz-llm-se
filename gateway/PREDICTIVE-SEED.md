# Predictive routing seed: shadow only

This branch validates and trains the handoff's 288-record controlled campaign.
It does not establish a production routing benefit. The HTTP adapter cannot
activate predictive routing: `create_shadow_app` accepts only `mode: shadow`.
The actual route remains sticky least-outstanding, with no retries or migration.

## Data and reproducibility

Base: `origin/profiling/bf16-fp8-request-records`, commit `4779289`.
Work branch: `codex/predictive-gateway`. The main checkout and open LaTeX summary
are preserved. Raw data is unchanged.

Run `profiling/validate_seed_campaign.py` with the full campaign root and a new
output file. It checks all twelve 24-row blocks, verified BF16/FP8 manifests,
matched trace bytes, source manifests, output controls, and pre-dispatch snapshot
ordering. Windows CRLF is normalized only when checking original LF trace hashes.
The audit also records hashes of the actual checked-out source files.

The builder now uses the enclosing prompt/session group rather than the logical
request ID containing a concurrency suffix. This keeps the same prompt together
across precisions, concurrency levels, and future turns. All 288 selected-worker
observations are used, with 200/52/36 train/validation/test rows in 50/13/9 groups.
No pilot, deprecated run, failure filtering, or graph outlier exclusion is added.

Example from the repository root (replace Python path for your platform):

```sh
python profiling/validate_seed_campaign.py \
  --root profiling/results/vast-full-bf16-fp8-2026-11-15 --out NEW/raw-validation.json
python profiling/build_predictive_router_dataset.py --runs \
  profiling/results/vast-full-bf16-fp8-2026-11-15/bf16/c1-r1 \
  profiling/results/vast-full-bf16-fp8-2026-11-15/bf16/c1-r2 \
  profiling/results/vast-full-bf16-fp8-2026-11-15/bf16/c1-r3 \
  profiling/results/vast-full-bf16-fp8-2026-11-15/bf16/c4-r1 \
  profiling/results/vast-full-bf16-fp8-2026-11-15/bf16/c4-r2 \
  profiling/results/vast-full-bf16-fp8-2026-11-15/bf16/c4-r3 \
  profiling/results/vast-full-bf16-fp8-2026-11-15/fp8/c1-r1 \
  profiling/results/vast-full-bf16-fp8-2026-11-15/fp8/c1-r2 \
  profiling/results/vast-full-bf16-fp8-2026-11-15/fp8/c1-r3 \
  profiling/results/vast-full-bf16-fp8-2026-11-15/fp8/c4-r1 \
  profiling/results/vast-full-bf16-fp8-2026-11-15/fp8/c4-r2 \
  profiling/results/vast-full-bf16-fp8-2026-11-15/fp8/c4-r3 --out NEW/dataset
python -m pip install -r profiling/requirements-training.txt
python profiling/train_regression.py --data NEW/dataset/training.jsonl --out NEW/model
```

Never reuse an existing output/model directory. No `--allow-unverified` was used.
Portable artifact: 15 observed allowlisted features, 64-tree forests, fixed seed
42, maximum depth 8, minimum leaf size 3, sklearn 1.7.2. No hyperparameters were
changed after viewing the test split. Latency labels never enter online features.

## Offline findings

Validation E2E MAE: 1751.66 ms versus 15404.66 ms constant-baseline MAE.
Test E2E MAE: 1635.57 ms versus 17708.62 ms constant-baseline MAE.
Validation TTFT MAE: 295.64 ms versus 1672.74 ms constant-baseline MAE.
Test TTFT MAE: 602.59 ms versus 1366.22 ms constant-baseline MAE.

These are prediction errors, not measured routing speedups. The success model is
constant because every seed observation succeeded. SLO labels are insufficient;
neither is a usable reliability gate. The dataset has one workload family,
forced outputs, limited load states, and different attention backends. Quality
is unmeasured, and prefix reuse was minimized rather than verified cold.

`stratified-errors.json` retains errors by precision, input bucket, output cap,
concurrency, and repetition. `seed_diagnostics.py` plots all observations and
144 descriptive FP8-minus-BF16 pairs. Paired evidence includes both dispatch
times, feature/load snapshots, and group/split identity. It cannot establish
what the other worker would have done at the same live queue state.

## Adapter boundary and tests

`gateway/policy.py` is a pure new-session policy. It validates portable artifacts,
checks known configurations, health, snapshot age/order, explicit quality approval,
required metrics, context/memory pressure, finite features/predictions, and feature
domain. Safe eligible candidates use minimum predicted E2E with optional predicted
TTFT/E2E SLO gates. Prediction failures use deterministic least-outstanding among
safe candidates; no safe candidate produces a rejection. A failed SLO gate rejects
rather than silently bypassing the SLO.

`gateway/shadow.py` collects completed pre-dispatch metrics and actual `/tokenize`
counts. Unknown request features remain missing. It does not infer output length,
quality, cached tokens, or turn count from the response. Snapshot source/times,
configuration hashes, exclusions, feature vectors and predictions are logged.
No request messages, generated content, or credentials enter decision logs.
Worker configuration provenance must be freshly verified before each deployment;
the example renderer's observed identities are not service discovery.

The baseline HTTP adapter pins before forwarding. Follow-up turns never consult
the model for reassignment; backend errors do not trigger cross-worker retries.
Its in-memory pins require one gateway process and do not survive restart.

Run mocked tests with:

```sh
python -m unittest discover -s tests -p 'test_predictive_policy.py' -v
python -m unittest discover -s tests -p 'test_shadow_gateway.py' -v
python -m unittest discover -s tests -p 'test_gateway.py' -v
python -m unittest discover -s profiling/tests -v
```

## Live canary and promotion stop conditions

Inspection on November 15 found both current Qwen3.8 engines using **float16**
compute and KV caches, not the historical campaign's BF16/FP8 pair:
SPARK03 workflow `b0245156-566a-4796-b5cc-1207514b4c95`, SPARK04 workflow
`cb9eae51-9223-4efc-a3c3-21cd49128559`. Both startup logs and specs confirm this.
The original gateway `4a8b8149-fb63-4dfa-9890-84736bcd3b1c` is preserved.

Isolated CPU-only canary: `9d2b6906-6151-4ec9-8f13-b2181e7490d4`, SPARK04,
port 19051, one CPU and 2 GiB, no GPU or model-volume mount. It correctly rejects
learned proposals for the live workers as `unknown_configuration`; quality is
also explicitly unapproved. Its baseline smoke is recorded separately from
prediction checks. No existing workflow was stopped or replaced.

The canary's model-list and telemetry calls succeeded, and its shadow policy
rejected both unknown configurations. Its first 8-token completion request to
node-a returned HTTP 504 after the gateway's 120-second read timeout. It did not
retry on node-b. Live two-session affinity and streaming therefore did **not**
pass. Fuzzball automatically terminated the canary after the smoke job failed;
it is errored, not a usable running endpoint. Failure logs are retained at
`fb://group/predictive-router/seed-2026-11-15-v1/canary-failure.log` (30-day TTL).
No second attempt or worker restart was made. The original workers and gateway
remain active. Upstream readiness/generation needs diagnosis before further
policy evaluation; a successful health/model-list response was insufficient.

Do not point production traffic at the canary or call it a mixed-precision router.
Promotion requires: restore/verify an approved mixed pair without disrupting
teammates; independently evaluate natural-output quality; collect real multi-turn
gateway traffic with balanced exposure and failures; lock SLOs and thresholds;
review shadow logs; and compare identical held-out baseline/predictive traces.
If quality, errors, or SLOs regress, keep the baseline. The current seed cannot
satisfy these requirements, even though offline prediction beats a constant.
