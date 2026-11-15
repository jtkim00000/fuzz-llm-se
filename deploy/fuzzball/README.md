# Nemotron stock worker on SPARK03

This is a deployment draft. The initial, unpinned workflow was
`eefd9cec-8003-4994-8c02-017a83f86e42`. Its staging ran on SPARK04, never
reached inference, and received a successful CLI stop response after SPARK03
was requested. Its scheduler allocations are now finished; the storage
allocation also reported that SPARK04 stopped responding. No working API URL,
runtime engine configuration, or benchmark results were obtained.

The current `nemotron-stock.fz` has **not** been submitted. It requires a
scheduler-visible placement label unique to CIQ-CHI-SPARK03:

```yaml
resource:
  annotations:
    nvidia.com/gpu.model: NVIDIA GB10
    ciq.com/worker-host: CIQ-CHI-SPARK03
  exclusive: true
```

`ciq.com/worker-host` is a proposed label, not an existing cluster feature.
The scheduling team must expose this label through the supported provisioner
configuration, or provide an existing equivalent selector. SPARK03 and SPARK04
currently share the `team2` provisioner and identical placement annotations.
The current login is not a cluster administrator. Do not submit until the
selector is available and placement scoring confirms eligibility. The draft
also refuses to start inference unless SPARK03's GPU UUID is visible:
`GPU-bb63ce4d-a7e3-9ccb-39c0-62427fe8fd8f`.

SPARK03 must be available for an exclusive allocation. A separate workflow
`5ad4d186-2c26-4435-bf0d-3f1a0806edfa` was running a CPU gateway there during
inspection. Coordinate with its owner; this deployment does not stop it.

The model is pinned to
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` revision
`6efb4a2a1c1fa277ce7b3df7a1416255011b1c99`. The NVIDIA serving image tag is
`nvcr.io/nvidia/vllm:25.12.post1-py3`, selected for ARM64 GB10. The image's
runtime version, commit, and OCI digest remain unverified. This stock baseline
does not override the KV-cache dtype. Its requested limits are 16,384 context
tokens, 4 sequences, and 2,048 batched tokens; effective settings must be read
from the running engine before populating the manifest.

After placement is available, validate and score the file through Fuzzball,
then start it. Do not regard a generated endpoint URL as a passed health check.
Fetch runtime provenance with `workflow_exec` from
`/data/profiling/runtime-provenance.json`, and save it locally. Since this is a
persistent service, cancellation is not a durable artifact export; publish
collected JSON/logs to `fb://user/` using `object_put` before cleanup.

The endpoint has user scope. Once it is live, use the existing authenticated
Fuzzball CLI to list endpoints and generate a short-lived endpoint token:

```sh
fuzzball workflow endpoints list -o json
fuzzball workflow endpoints generate-token ENDPOINT_ID --expiration 1h
```

Handle the token through the team's approved credential channel. Keep the
bearer value in a file with mode `0600` under the ignored `.local-inference/`
directory; do not put it in the manifest, Git, or shared logs. The checker reads
that file and never logs authentication headers:

```sh
.local-inference/venv/bin/python deploy/fuzzball/check_worker_endpoint.py \
  --endpoint https://ACTUAL-ENDPOINT/ \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --token-file .local-inference/remote-worker/endpoint-token \
  --output results/nemotron-spark03-checks
```

It checks `/health`, the exact `/v1/models` name, `/metrics`, a chat completion,
and streamed completion with `[DONE]`. It makes two inference requests with
thinking disabled. These are connectivity checks, not performance benchmarks.
Only after these pass and the scheduling team confirms the quiet window should
the 30-minute benchmark window begin.

`fetch_profiling_metadata.py` downloads only the pinned tokenizer/config files
and queries public registry metadata. The first tokenizer download attempt
timed out before any file was saved. No checkpoint weights were downloaded to
the Mac. A successful registry lookup would identify the tag's current digest;
it would still need to be reconciled with the image actually used at runtime.
