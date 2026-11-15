# Local worker endpoints for router tests

The Mac setup exposes two HTTP worker facades backed by the existing Llama
server. Both share one model process, GPU queue, and KV cache. Model weights
remain the downloaded 4-bit Llama 3.1 8B Instruct conversion.

| Worker | API base | Requested KV cache | Actual KV cache | Simulation |
| --- | --- | --- | --- | --- |
| BF16 | `http://127.0.0.1:8001/v1` | BF16 | BF16 | No precision substitution |
| FP8 test stand-in | `http://127.0.0.1:8002/v1` | FP8 | BF16 | Explicitly simulated |

The installed vLLM Metal 0.33.0 rejects `--kv-cache-dtype fp8`. The FP8 facade
exists to exercise routing, streaming, session headers, and error handling. It
does not quantize caches or emulate FP8 output differences, memory savings,
latency, or throughput. Shared caches also mean this setup cannot validate
cache isolation or the performance benefit of session affinity. Use distinct
GPU-backed vLLM workers for precision and fleet benchmarks.

## Start, inspect, and stop

The local model and Python environment were downloaded into `.local-inference/`.
If the model server on port 8000 is stopped, run this in a terminal:

```bash
bash .local-inference/serve.sh
```

Start the two facades in the background:

```bash
.local-inference/venv/bin/python scripts/local_workers.py start
.local-inference/venv/bin/python scripts/local_workers.py status
```

The controller leaves other applications and the underlying Llama server
untouched. Worker PID files and logs are under `.local-inference/workers/`.
Stop only the two facades with:

```bash
.local-inference/venv/bin/python scripts/local_workers.py stop
```

For a foreground worker, use one of these commands per terminal. FP8 simulation
must be explicitly enabled:

```bash
.local-inference/venv/bin/python -m local_workers --worker bf16
.local-inference/venv/bin/python -m local_workers --worker fp8 --simulate-fp8
```

The facade dependencies are pinned in `requirements-local-workers.txt` and
already exist in the downloaded environment. A fresh environment needs these
dependencies plus a separately configured inference backend.

## Call either worker

Both use the same model alias, `llama-3.1-8b-local`.

```bash
curl -i http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/worker-info

curl -i http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: repo-session-1' \
  -d '{"model":"llama-3.1-8b-local","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":32,"temperature":0}'
```

Health, worker-info, and forwarded responses identify the worker and actual precision through
`X-Worker-ID`, `X-KV-Cache-Requested`, `X-KV-Cache-Actual`,
`X-Precision-Simulated`, `X-Test-Only`, and `X-Shared-Backend` headers.
`/worker-info` reports the same information as JSON. `/health` returns 503
when the shared backend is unavailable. Inference connection failures return
502; upstream connection/read timeouts before response headers return 504.
Upstream model errors and SSE streams are forwarded without body changes.
Session IDs are forwarded; the facades do not implement a session router.
The `/metrics` endpoint forwards the shared backend's metrics, so those values
are not independent worker measurements.

## Verify

```bash
.local-inference/venv/bin/python -m unittest discover -s tests -p 'test_local_workers.py' -v

.local-inference/venv/bin/python .local-inference/test_endpoints.py \
  --base-url http://127.0.0.1:8001 \
  --report .local-inference/workers/bf16-endpoint-results.json

.local-inference/venv/bin/python .local-inference/test_endpoints.py \
  --base-url http://127.0.0.1:8002 \
  --report .local-inference/workers/fp8-simulated-endpoint-results.json
```

The two root URLs in `.env.example` are configuration examples for a future
router. The facades' current default backend is `http://127.0.0.1:8000`; neither
the controller nor Fuzzball automatically reads that example environment file.
