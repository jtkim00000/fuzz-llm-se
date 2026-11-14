# Gateway serving prototype

Install with `python -m pip install -r gateway/requirements.txt`, then run
`uvicorn gateway.app:app_from_env --factory --host 127.0.0.1 --port 8080`.
Set `GATEWAY_BF16_URL` and
`GATEWAY_FP8_URL` to worker HTTP(S) origins (without `/v1`). Defaults are
localhost ports 8001 and 8002, matching the team's `fuzzball_config` worker
facades; that branch must be started separately. Optional
`GATEWAY_WORKER_TOKEN` is sent to both workers and should be supplied via a
secret, never committed. The gateway does not forward the caller's bearer token.

For a single local Ollama worker, set `GATEWAY_LOCAL_OLLAMA_URL` to
`http://127.0.0.1:11434` before starting the gateway. This disables the two
precision-labelled worker configuration: it is a single development worker,
not a BF16/FP8 comparison. Ollama must be running and have a model installed.

```sh
GATEWAY_LOCAL_OLLAMA_URL=http://127.0.0.1:11434 \
  .venv-aiperf/bin/uvicorn gateway.app:app_from_env --factory --host 127.0.0.1 --port 18080
```

Then check `curl http://127.0.0.1:18080/v1/models`. A chat request must include
`X-Session-ID`, for example:

```sh
curl -N http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Session-ID: example-session' \
  -d '{"model":"llama3.1:8b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

POST `/v1/chat/completions` with `X-Session-ID` and an OpenAI-compatible body.
Responses preserve upstream status/body and stream bytes; `X-Routed-Node` and
`X-Policy` identify the selection. GET `/health` checks only the gateway.
New sessions use fewest in-flight requests with round-robin ties; sessions stay
pinned even after upstream failures to avoid silently discarding cached context.
`/v1/models` forwards the first worker's model list; `/health` does not check
worker readiness. No automatic health probing, retries, failover, prediction,
session expiry, distributed session persistence, or authentication is provided.
Run a single gateway process; restart loses affinity. Protect the endpoint and
provide worker health/auth integration before exposing it to a network.
Session IDs are currently global rather than scoped to an authenticated user;
concurrent turns in one session are not serialized. There is no durable
per-request logging or health-aware failover yet.

Run tests with `python -m unittest discover -s tests -p 'test_gateway.py' -v`.
Future prediction integration should receive only pre-assignment request
features and worker state and return estimated costs; it must preserve existing
sessions and fail back to this baseline if predictions are unavailable.
