#!/bin/sh
# Full matched fallback campaign: 144 measured requests per precision.
set -eu
: "${ENGINE:?bf16 or fp8}"
PY=/results/benchmark-env-xzAhY13Y/venv/bin/python
TOOLS=/opt/profiling
BASE=/results/vast-full-bf16-fp8-2026-11-15
MODEL=/models/nvidia--Qwen3.8-27B-NVFP4/482ca0f3832238542f8f5295dde86b5f22711d80
SRC=$(find /datasets -name ReactomePathways.txt -type f | head -n 1)
test -x "$PY"; test -n "$SRC"
if test "$ENGINE" = bf16; then URL=http://192.168.99.13:18027; KV=bfloat16; WORKFLOW=52626fae-3879-4559-85c0-d2de4a124f96; BACKEND=FLASH_ATTN
elif test "$ENGINE" = fp8; then URL=http://192.168.99.12:18027; KV=fp8; WORKFLOW=346f74e3-9826-45dc-adab-6bc95067fcc4; BACKEND=FLASHINFER
else exit 2; fi
ROOT="$BASE/$ENGINE"; rm -rf "$ROOT"; mkdir -p "$ROOT/preflight" "$ROOT/manifests"
cat > "$ROOT/manifests/worker.json" <<EOF
{"effective_configuration_verified":true,"fuzzball_workflow_id":"$WORKFLOW","worker_id":"$ENGINE","engine_version":"vLLM 0.28.0","model_checkpoint":"nvidia/Qwen3.8-27B-NVFP4","model_revision":"482ca0f3832238542f8f5295dde86b5f22711d80","tokenizer_revision":"482ca0f3832238542f8f5295dde86b5f22711d80","served_model_id":"nvidia/Qwen3.8-27B-NVFP4","effective_kv_dtype":"$KV","attention_backend":"$BACKEND","max_model_len":8192,"batching_settings":{"max_num_seqs":4,"max_num_batched_tokens":2048,"gpu_memory_utilization":0.6,"kv_cache_memory_bytes":8589934592,"enforce_eager":true},"prefix_caching":true,"warmup_and_cache_procedure":"separate preflight plus short/long warmup; prefix-reuse-minimized, not verified cold","missing_measurements":["KV scales","GPU exporter"],"context_limit_decision":"fallback 7168 because both verified workers have max_model_len 8192"}
EOF
"$PY" "$TOOLS/build_vast_compact_campaign.py" --source "$SRC" --tokenizer "$MODEL" --out "$ROOT/traces" --prompts-per-cell 3 --repetitions 3 --campaign vast-bf16-fp8-full-fallback-8192
curl -fsS "$URL/v1/models" > "$ROOT/preflight/models.json"
curl -fsS -N -H 'Content-Type: application/json' -d '{"model":"nvidia/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"NONCE: readiness-full-20261115\nWrite a catalogue entry."}],"stream":true,"stream_options":{"include_usage":true},"temperature":0,"max_tokens":32,"min_tokens":32,"ignore_eos":true,"chat_template_kwargs":{"enable_thinking":false}}' "$URL/v1/chat/completions" > "$ROOT/preflight/forced-32.sse"
for SHAPE in short long; do
  "$PY" -c 'import json,sys; rows=[json.loads(x)["turns"][0] for x in open(sys.argv[1])]; row=(min if sys.argv[2]=="short" else max)(rows,key=lambda r:r["input_bucket"]); print(json.dumps({"model":"nvidia/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":row["text"]}],"stream":True,"stream_options":{"include_usage":True},"temperature":0,"max_tokens":8,"min_tokens":8,"ignore_eos":True,"chat_template_kwargs":{"enable_thinking":False}}))' "$ROOT/traces/c1-r1.jsonl" "$SHAPE" | curl -fsS -N -H 'Content-Type: application/json' --data-binary @- "$URL/v1/chat/completions" > "$ROOT/preflight/warmup-$SHAPE.sse"
done
for C in 1 4; do
  for R in 1 2 3; do
    BLOCK=c$C-r$R
    cat > "$ROOT/config-$BLOCK.json" <<EOF
{"policy":"single","concurrency":$C,"sample_interval_seconds":0.2,"metrics_timeout_seconds":3,"request_timeout_seconds":900,"cache_condition":"prefix-reuse-minimized, not verified cold","workers":[{"id":"$ENGINE","url":"$URL","metrics_url":"$URL/metrics","health_url":"$URL/health","model":"nvidia/Qwen3.8-27B-NVFP4","manifest_path":"$ROOT/manifests/worker.json","metric_map":{"running_requests":{"metric":"vllm:num_requests_running"},"queued_requests":{"metric":"vllm:num_requests_waiting"},"kv_cache_usage":{"metric":"vllm:kv_cache_usage_perc"}}}]}
EOF
    "$PY" "$TOOLS/collect_training.py" --config "$ROOT/config-$BLOCK.json" --trace "$ROOT/traces/$BLOCK.jsonl" --out "$ROOT/$BLOCK" --tokenizer "$MODEL"
    "$PY" "$TOOLS/normalize_canonical_records.py" --run-dir "$ROOT/$BLOCK" --run-id "full-vast-$ENGINE-$BLOCK-2026-11-15"
  done
done
echo "ARTIFACT_ROOT=$ROOT"
