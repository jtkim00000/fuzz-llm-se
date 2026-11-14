#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="results/local-smoke-$run_id"
mkdir -p "$output_dir"
git rev-parse HEAD > "$output_dir/git-commit.txt"
.venv-aiperf/bin/aiperf profile \
  --model llama3.1:8b --url http://127.0.0.1:11434 \
  --endpoint-type chat --streaming --tokenizer builtin \
  --use-legacy-max-tokens --use-server-token-count \
  --concurrency 1 --request-count 5 --warmup-request-count 1 \
  --isl 128 --osl 64 --no-server-metrics \
  --artifact-dir "$output_dir" --ui none
