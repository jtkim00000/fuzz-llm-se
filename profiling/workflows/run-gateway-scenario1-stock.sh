#!/bin/sh
# Scenario 1: two stock-vLLM workers behind the least-outstanding gateway.
set -eu
PY=/results/benchmark-env-xzAhY13Y/venv/bin/python
TOOLS=/opt/profiling
MODEL=/models/nvidia--Qwen3.8-27B-NVFP4/482ca0f3832238542f8f5295dde86b5f22711d80
TRACE_ROOT=/results/vast-full-bf16-fp8-2026-11-15/bf16/traces
ROOT=/results/gateway-scenario1-stock-2026-11-15
GATEWAY=http://192.168.99.13:19038
test -x "$PY"; test -d "$TRACE_ROOT"; test -d "$MODEL"
rm -rf "$ROOT"; mkdir -p "$ROOT/preflight" "$ROOT/manifests"
cat > "$ROOT/manifests/gateway.json" <<EOF
{"scenario":1,"gateway_policy":"least_outstanding_requests","gateway_url":"$GATEWAY","gateway_worker_aliases":{"bf16":"http://192.168.99.12:18027","fp8":"http://192.168.99.13:18027"},"actual_worker_configuration":"stock dtype auto; KV cache dtype auto","worker_workflows":{"stock_a":"00dfc1a3-4d2f-45c2-8585-ebe359a04571","stock_b":"b1b2876e-4ecd-40d8-8574-9ea9fb9bf30c"},"model_checkpoint":"nvidia/Qwen3.8-27B-NVFP4","model_revision":"482ca0f3832238542f8f5295dde86b5f22711d80","max_model_len":8192,"max_num_seqs":4,"max_num_batched_tokens":2048,"gpu_memory_utilization":0.6,"kv_cache_memory_bytes":8589934592,"effective_configuration_verified":true,"telemetry_limitations":["gateway exposes no per-worker Prometheus running/waiting counters","gateway response header records selected alias"]}
EOF
curl -fsS "$GATEWAY/health" > "$ROOT/preflight/gateway-health.json"
curl -fsS "$GATEWAY/state" > "$ROOT/preflight/gateway-state-before.json"
curl -fsS "$GATEWAY/v1/models" > "$ROOT/preflight/models.json"
curl -fsS -N -H 'Content-Type: application/json' -H 'X-Session-ID: scenario1-warmup-20261115' -d '{"model":"nvidia/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"NONCE: scenario1 gateway warmup\nWrite a catalogue entry."}],"stream":true,"stream_options":{"include_usage":true},"temperature":0,"max_tokens":32,"min_tokens":32,"ignore_eos":true,"chat_template_kwargs":{"enable_thinking":false}}' "$GATEWAY/v1/chat/completions" > "$ROOT/preflight/forced-32.sse"
for SHAPE in short long; do
  "$PY" -c 'import json,sys; rows=[json.loads(x)["turns"][0] for x in open(sys.argv[1])]; row=(min if sys.argv[2]=="short" else max)(rows,key=lambda r:r["input_bucket"]); print(json.dumps({"model":"nvidia/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":row["text"]}],"stream":True,"stream_options":{"include_usage":True},"temperature":0,"max_tokens":8,"min_tokens":8,"ignore_eos":True,"chat_template_kwargs":{"enable_thinking":False}}))' "$TRACE_ROOT/c1-r1.jsonl" "$SHAPE" | curl -fsS -N -H 'Content-Type: application/json' -H "X-Session-ID: scenario1-warmup-$SHAPE-20261115" --data-binary @- "$GATEWAY/v1/chat/completions" > "$ROOT/preflight/warmup-$SHAPE.sse"
done
for C in 1 4; do
  for R in 1 2 3; do
    BLOCK=c$C-r$R
    cp "$TRACE_ROOT/$BLOCK.jsonl" "$ROOT/$BLOCK-trace-source.jsonl"
    cat > "$ROOT/config-$BLOCK.json" <<EOF
{"policy":"single","concurrency":$C,"sample_interval_seconds":0.2,"metrics_timeout_seconds":3,"request_timeout_seconds":900,"cache_condition":"gateway-path; prefix cache state unverified","workers":[{"id":"gateway","url":"$GATEWAY","health_url":"$GATEWAY/health","model":"nvidia/Qwen3.8-27B-NVFP4","manifest_path":"$ROOT/manifests/gateway.json"}]}
EOF
    "$PY" "$TOOLS/collect_training.py" --config "$ROOT/config-$BLOCK.json" --trace "$TRACE_ROOT/$BLOCK.jsonl" --out "$ROOT/$BLOCK" --tokenizer "$MODEL"
    "$PY" "$TOOLS/normalize_canonical_records.py" --run-dir "$ROOT/$BLOCK" --run-id "scenario1-stock-gateway-$BLOCK-2026-11-15"
  done
done
curl -fsS "$GATEWAY/state" > "$ROOT/gateway-state-after.json"
echo "ARTIFACT_ROOT=$ROOT"
