#!/bin/sh
# Emit measured-request progress from the shared Fuzzball results volume.
set -eu
ROOT=/results/gateway-scenario4-predictive-bf16-fp8-2026-11-15
TOTAL=144
announced=0
while :; do
  count=0
  for f in "$ROOT"/c*-r*/requests.jsonl; do
    test -f "$f" || continue
    lines=$(wc -l < "$f")
    count=$((count + lines))
  done
  if test "$count" -eq 0; then
    if test "$announced" -eq 0; then
      echo "PHASE=awaiting-first-measured-record total=$TOTAL"
      announced=1
    fi
  else
    echo "MEASURED_PROMPTS=$count/$TOTAL remaining=$((TOTAL - count))"
  fi
  if test -f "$ROOT/gateway-state-after.json"; then
    echo "SCENARIO4_BENCHMARK_COMPLETE measured_prompts=$count/$TOTAL"
    exit 0
  fi
  sleep 15
done
