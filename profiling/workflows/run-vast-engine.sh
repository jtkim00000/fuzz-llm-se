#!/bin/sh
set -eu
: "${ENGINE:?set ENGINE to bf16 or fp8}"
PYTHON=/results/benchmark-env-xzAhY13Y/venv/bin/python
TOOLS=/opt/profiling
BASE=/results/profiling-bf16-fp8-vast-2026-11-14
ROOT="$BASE/$ENGINE"
MODEL=/models/nvidia--Qwen3.8-27B-NVFP4/482ca0f3832238542f8f5295dde86b5f22711d80
SOURCE=$(find /datasets -name ReactomePathways.txt -type f | head -n 1)
test -n "$SOURCE"; test -x "$PYTHON"
rm -rf "$ROOT"; mkdir -p "$ROOT/manifests" "$ROOT/preflight"
if test "$ENGINE" = bf16; then
  URL=http://192.168.99.12:18027; WORKER=SPARK03-bf16-worker-a; WORKFLOW=c8b096fd-97a8-44f2-91bd-7d60db53c916; KV=bfloat16; BACKEND=FlashAttention; LAUNCH='--kv-cache-dtype bfloat16; effective value verified from vLLM startup log'
elif test "$ENGINE" = fp8; then
  URL=http://192.168.99.13:18027; WORKER=SPARK04-fp8-worker; WORKFLOW=95e2dc41-c219-4cde-8fb3-64858ca262e4; KV=fp8_e4m3fn; BACKEND='FlashInfer; decode backend xqa'; LAUNCH='--kv-cache-dtype fp8; resolved torch.float8_e4m3fn verified from vLLM startup log'
else echo "Unknown ENGINE: $ENGINE" >&2; exit 2; fi
cat > "$ROOT/manifests/worker.json" <<EOF
{"effective_configuration_verified":true,"fuzzball_workflow_id":"$WORKFLOW","worker_id":"$WORKER","engine_version":"vLLM 0.28.0","image":"vllm/vllm-openai:v0.28.0","model_checkpoint":"nvidia/Qwen3.8-27B-NVFP4","model_revision":"482ca0f3832238542f8f5295dde86b5f22711d80","tokenizer_revision":"482ca0f3832238542f8f5295dde86b5f22711d80","served_model_id":"nvidia/Qwen3.8-27B-NVFP4","effective_kv_dtype":"$KV","kv_scales":"not reported in launch logs","attention_backend":"$BACKEND","max_model_len":8192,"batching_settings":{"max_num_seqs":4,"max_num_batched_tokens":2048,"gpu_memory_utilization":0.6,"kv_cache_memory_bytes":8589934592,"enforce_eager":true},"prefix_caching":true,"thinking_mode":false,"gpu":"NVIDIA GB10","gpu_count":1,"warmup_and_cache_procedure":"One preflight request per worker, then measured closed-loop concurrency=4; server cache state is not resettable from the benchmark client.","launch_arguments":"$LAUNCH","client_to_worker_route":"$URL","hardware_metrics_endpoint":"$URL/metrics","missing_measurements":["KV scale values","GPU counter exporter"],"notes":["Weights use modelopt_mixed NVFP4; this is distinct from attention KV dtype."]}
EOF
cat > "$ROOT/config.json" <<EOF
{"policy":"single","concurrency":4,"sample_interval_seconds":1,"metrics_timeout_seconds":3,"request_timeout_seconds":900,"cache_condition":"unverified","workers":[{"id":"$ENGINE","url":"$URL","metrics_url":"$URL/metrics","health_url":"$URL/health","model":"nvidia/Qwen3.8-27B-NVFP4","manifest_path":"$ROOT/manifests/worker.json","metric_map":{"running_requests":{"metric":"vllm:num_requests_running"},"queued_requests":{"metric":"vllm:num_requests_waiting"},"kv_cache_usage":{"metric":"vllm:kv_cache_usage_perc"}}}]}
EOF
"$PYTHON" "$TOOLS/build_vast_reactome_trace.py" --source "$SOURCE" --tokenizer "$MODEL" --out "$ROOT/vast-reactome-trace.jsonl" --count 100 --min-input-tokens 5200 --max-input-tokens 6200 --output-limit 128
curl -sS "$URL/v1/models" > "$ROOT/preflight/models.json"
curl -sS -N -H 'Content-Type: application/json' -d '{"model":"nvidia/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"Reply with exactly: ready"}],"max_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true},"chat_template_kwargs":{"enable_thinking":false}}' "$URL/v1/chat/completions" > "$ROOT/preflight/stream.sse"
"$PYTHON" "$TOOLS/collect_training.py" --config "$ROOT/config.json" --trace "$ROOT/vast-reactome-trace.jsonl" --out "$ROOT/measurement" --tokenizer "$MODEL"
"$PYTHON" "$TOOLS/normalize_canonical_records.py" --run-dir "$ROOT/measurement" --run-id "vast-reactome-100-c4-$ENGINE-2026-11-14"
echo "ARTIFACT_ROOT=$ROOT"
