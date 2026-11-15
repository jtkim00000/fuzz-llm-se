#!/bin/sh
# Scenario 4: one verified BF16-KV worker and one verified FP8-KV worker,
# selected by the live predictive, session-sticky PrecisionSplit gateway.
set -eu

PY=/results/benchmark-env-xzAhY13Y/venv/bin/python
TOOLS=/opt/profiling
MODEL=/models/nvidia--Qwen3.8-27B-NVFP4/482ca0f3832238542f8f5295dde86b5f22711d80
TRACE_ROOT=/results/vast-full-bf16-fp8-2026-11-15/bf16/traces
ROOT=/results/gateway-scenario4-predictive-bf16-fp8-2026-11-15
GATEWAY=http://192.168.99.13:19039

test -x "$PY"; test -d "$TRACE_ROOT"; test -d "$MODEL"
if test -e "$ROOT"; then
  echo "Refusing to overwrite existing artifact root: $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT/preflight" "$ROOT/manifests"

cat > "$ROOT/manifests/gateway.json" <<'EOF'
{"scenario":4,"gateway_policy":"minimum_predicted_completion_time","gateway_url":"http://192.168.99.13:19039","gateway_workflow":"49774fbc-3324-4b23-b9f7-f5bf46a17db9","prediction_model_version":"2a9372b996c76518158cd091e1fba9366d265bcf3d8791e0039108eee05c89d5","prediction_model_error":null,"worker_workflows":{"bf16":"814f4652-f73c-4cd2-99d5-a141e7bf8c61","fp8":"50fbe486-0cc2-42d0-906a-3db329f0451d"},"worker_urls":{"bf16":"http://192.168.99.13:18027","fp8":"http://192.168.99.12:18027"},"effective_worker_configuration":{"bf16":{"model_dtype":"bfloat16","kv_cache_dtype":"bfloat16","attention_backend":"FLASH_ATTN","kv_cache_tokens":80474},"fp8":{"model_dtype":"bfloat16","kv_cache_dtype":"torch.float8_e4m3fn","attention_backend":"FLASHINFER","kv_cache_tokens":114005}},"model_checkpoint":"nvidia/Qwen3.8-27B-NVFP4","model_revision":"482ca0f3832238542f8f5295dde86b5f22711d80","max_model_len":8192,"max_num_seqs":4,"max_num_batched_tokens":2048,"kv_cache_memory_bytes":8589934592,"effective_configuration_verified":true,"warmup":"No additional benchmark-client generation warmups: live worker logs show engine/JIT warmup and gateway validation traffic completed before this measured run.","telemetry_limitations":["Gateway response header records selected worker","gateway has no per-worker Prometheus snapshot in this collection"]}
EOF

# Read-only readiness evidence. The live services have already completed
# engine/JIT warm-up; do not add unmeasured generation traffic here.
curl -fsS "$GATEWAY/health" > "$ROOT/preflight/gateway-health.json"
curl -fsS "$GATEWAY/state" > "$ROOT/preflight/gateway-state-before.json"
curl -fsS "$GATEWAY/v1/models" > "$ROOT/preflight/models.json"

for C in 1 4; do
  for R in 1 2 3; do
    BLOCK=c$C-r$R
    cp "$TRACE_ROOT/$BLOCK.jsonl" "$ROOT/$BLOCK-trace-source.jsonl"
    cat > "$ROOT/config-$BLOCK.json" <<EOF
{"policy":"single","concurrency":$C,"sample_interval_seconds":0.2,"metrics_timeout_seconds":3,"request_timeout_seconds":900,"cache_condition":"gateway-path; pre-existing engine warm-up; prefix cache state unverified","workers":[{"id":"predictive_gateway","url":"$GATEWAY","health_url":"$GATEWAY/health","model":"nvidia/Qwen3.8-27B-NVFP4","manifest_path":"$ROOT/manifests/gateway.json"}]}
EOF
    "$PY" "$TOOLS/collect_training.py" --config "$ROOT/config-$BLOCK.json" --trace "$TRACE_ROOT/$BLOCK.jsonl" --out "$ROOT/$BLOCK" --tokenizer "$MODEL"
    "$PY" "$TOOLS/normalize_canonical_records.py" --run-dir "$ROOT/$BLOCK" --run-id "scenario4-predictive-bf16-fp8-gateway-$BLOCK-2026-11-15"
  done
done

curl -fsS "$GATEWAY/state" > "$ROOT/gateway-state-after.json"
echo "ARTIFACT_ROOT=$ROOT"
