# PrecisionSplit local prototype

Two independent mock HTTP workers and one gateway. No model weights, GPU,
vLLM, or Fuzzball are needed. BF16/FP8 are simulation labels; the delays are
invented demo parameters, not measurements of precision performance.

From the repository root, install the shared Python dependencies:

```sh
uv venv --python 3.12 .venv-precisionsplit
uv pip install --python .venv-precisionsplit/bin/python -r requirements-precisionsplit.txt
```

Start each command in its own terminal:

```sh
# Terminal 1
.venv-precisionsplit/bin/python -m precision_split worker --mode bf16 --port 9101
# Terminal 2
.venv-precisionsplit/bin/python -m precision_split worker --mode fp8 --port 9102
# Terminal 3
.venv-precisionsplit/bin/python -m precision_split gateway --port 9100 \
  --bf16-url http://127.0.0.1:9101 --fp8-url http://127.0.0.1:9102
```

All processes bind to localhost by default. Ports, host, worker URLs, and
gateway worker timeout are configurable through `--help`. Use one Uvicorn
process per service: queues, load reservations, and session pins are in memory.

Run the demo with idle workers and the supplied default timing rules:

```sh
.venv-precisionsplit/bin/python -m precision_split.demo \
  --gateway-url http://127.0.0.1:9100 --output results/precisionsplit-demo.json
```

It asserts and reports:

1. An idle short request selects BF16: 0.100 s versus FP8's 0.165 s.
2. A longer request occupies BF16; a new short session selects FP8 because
   its predicted finish is sooner.
3. A follow-up to the first session stays on BF16 and waits in its FIFO queue,
   despite FP8 being idle. All reservations are released afterward.

The busy request and follow-up run concurrently from the client. The demo
waits for observable assignment/worker state rather than guessing startup
delays. Reports include both scores, decisions, worker identities and queue
waits. Assignment/release events are also JSON logs in the gateway terminal.

Send your own requests:

```sh
curl -s http://127.0.0.1:9100/infer -H 'Content-Type: application/json' \
  -d '{"session_id":"chat-1","input_length":10,"expected_output_length":10}'
curl -s http://127.0.0.1:9100/state
```

Lengths are synthetic token counts, supplied explicitly; no tokenizer runs.
Responses return the session ID, routing scores and dummy worker response.
Omitting `session_id` creates a new session; reuse the returned ID for affinity.
Workers expose `/health`, `/state`, and `/infer`; direct worker `/infer` requests
also require a unique `request_id`. Direct traffic bypasses gateway accounting.
Gateway `/health` checks its own process, not backend availability.

## Timing and routing

All three processes read `precision_split/timings.json`. To change delays,
pass the **same** `--timings /path/to/timings.json` to each process and restart:

`service = base_seconds + input_seconds_per_token * input_length + output_seconds_per_token * expected_output_length`

The gateway scores each worker with its estimated remaining assigned work plus
the new service estimate. It subtracts elapsed time from the estimated running
head, includes every queued reservation in full, and starts the next estimate
when the head releases. Network overhead is not modeled. Worker FIFO order is
HTTP arrival order; independent HTTP arrivals can differ from assignment order.

One async lock covers scoring, deterministic selection, session pinning and
reservation insertion. Equal scores choose the alphabetically first worker
(`bf16`). Forwarding occurs after reservation. Follow-ups reserve load too,
even when affinity chooses the slower finish. Completion, upstream errors,
timeouts and cancellation release the reservation in `finally`.

HTTP forwarding lives in `forwarding.py`; the routing ledger is in `routing.py`.
Failures return 502/504, without automatic retries. A failed session remains
pinned. A timeout can leave actual upstream work running after the reservation
is released; real deployment needs cancellation or reconciliation with telemetry.
Session pins and reservations are lost on gateway restart.

Run tests:

```sh
.venv-precisionsplit/bin/python -m unittest discover -s tests -v
```

## Later deployment

Replace the mock `/infer` adapter in `forwarding.py` with requests to two
Fuzzball-hosted vLLM `/v1/chat/completions` endpoints using the same pinned
model/build, one with BF16 KV and one with supported, verified FP8 KV. Add
approved authentication, streaming and failure handling there. Replace timing
coefficients with matched benchmark estimates, then reconcile reservations
with live worker queue/cache/load signals. Keep session affinity. No cluster
workflow or existing serving configuration is changed by this prototype.
