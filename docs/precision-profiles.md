# BF16 and FP8 KV-cache workers

Use one stock vLLM build with two launch profiles. Each worker runs its own
engine process and GPU allocation. A Git branch does not select the runtime
dtype: the launch configuration does. Keeping both profiles together avoids
unrelated engine changes becoming extra differences in the comparison.

| Setting | BF16 worker | FP8 worker |
| --- | --- | --- |
| Profile | `configs/kv-cache/bf16.yaml` | `configs/kv-cache/fp8.yaml` |
| Model compute dtype | `bfloat16` | `bfloat16` |
| Requested attention KV-cache dtype | `bfloat16` | `fp8` |
| Checkpoint and revision | Same pinned checkpoint | Same pinned checkpoint |
| Mamba state configuration | Same settings | Same settings |

These profiles select existing engine paths. They do not convert every kernel,
accumulator, model weight, or Mamba state to FP8. The vendored engine's Mamba
cache dtype options do not include FP8. A checkpoint with NVFP4 weights keeps
its quantization; `dtype: bfloat16` does not create a BF16 checkpoint from it.
FP8 weight quantization is a separate experiment.

## Launch on the GB10 workers

Use the same completed, pinned model snapshot on both workers. The existing
BF16 download and the proposal's NVFP4 checkpoint are different checkpoints;
choose one for the matched pair and record which one was used. Run these
commands on the GPU nodes from a checkout containing the profiles:

```sh
MODEL_DIR='/data/models/YOUR_COMPLETED_PINNED_SNAPSHOT'

# Worker A, on the first GPU node:
vllm serve "$MODEL_DIR" --config configs/kv-cache/bf16.yaml

# Worker B, on the second GPU node:
vllm serve "$MODEL_DIR" --config configs/kv-cache/fp8.yaml
```

The examples show the precision configuration only. Keep the approved Nemotron
parser options, served model name, context limit, batching, GPU memory
allocation, prefix caching, and thinking mode identical. Separate nodes can
both use port 8000. Record the actual endpoints returned by Fuzzball.

CLI arguments override YAML values. Remove conflicting `--dtype` or
`--kv-cache-dtype` overrides when attaching a profile to an existing command.

## Attach a profile to a Fuzzball service

A remote container does not automatically contain this checkout. Bind the
profile into the worker and add `--config /opt/kv-profile.yaml` to its existing
`vllm serve` command. Merge these entries into the existing `files` mappings:

```yaml
files:
  fp8-kv-profile.yaml: |
    dtype: bfloat16
    kv-cache-dtype: fp8
services:
  vllm:
    files:
      /opt/kv-profile.yaml: file://fp8-kv-profile.yaml
```

This is a fragment, not a complete workflow. It does not restart the current
worker or change a running download. Relaunch a worker through the team's
scheduling procedure to change its cache dtype; preserve or reuse the
downloaded snapshot rather than submitting a duplicate model ingress.
The stock-baseline workflow intentionally has no KV override.

Editing vendored source does not change an external prebuilt image. Custom
kernel experiments require building and deploying an image from the modified
commit; these stock KV profiles do not require a custom kernel build.

## Verify the actual engine

Check the deployed version's `vllm serve --help` for dtype support. Confirm the
resolved dtype and any skipped layers from engine logs or instrumentation,
then test health, model listing, and streaming output. The Mac Metal test
facades cannot validate GPU FP8 execution.

Record whether FP8 KV scales come from checkpoint calibration, supported
runtime calibration, or defaults. NVFP4 weight scales do not establish that
KV scales are calibrated. Calibration flags vary by engine version, so these
profiles do not hard-code one. Record the attention backend and effective
Mamba state dtypes separately.

Run identical correctness tasks and matched cold/warm traces. Keep separate
result directories and record the profile, engine image digest, checkpoint
revision, scale source, and effective dtypes. Follow the
[precision comparison guide](02-kv-precision-comparison.md).

References: [vLLM serve configuration](https://docs.vllm.ai/en/stable/cli/serve/)
and [FP8 KV-cache configuration and calibration](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/).
