# PrecisionSplit: least-outstanding-request routing

On the `greedy-gateway` branch, new sessions go to the healthy worker with
fewer outstanding requests. This applies to both the local mock gateway and
`vllm-gateway`, through the shared `GreedyRouter` implementation.

Outstanding means every request assigned through this gateway that has not
finished: running, waiting upstream, and streaming. It does not mean lifetime
requests or number of sessions. For example, with BF16 handling 3 requests
and FP8 handling 1, a new session selects FP8. Equal counts rotate in sorted
worker order, starting with BF16, so sequential requests also spread evenly.

Selection and increment happen under one lock before HTTP forwarding.
Completion, failure and cancellation release the reservation. Streams remain
counted until closure. A long-running request still counts even if its old
timing estimate has expired. Follow-ups stay on their session's worker and
increment that worker's count. This affinity can intentionally override balance.
The real gateway excludes unhealthy workers for new sessions and returns 503
when a pinned worker is unavailable.

Timing coefficients do not select workers on this branch. They still configure
mock processing delays, and timing estimates remain in diagnostics for
compatibility. Decisions log both counts and `least_outstanding_requests` or
`session_affinity`; `/state` reports `routing_policy` and current counts.

This balances request counts, not tokens, GPU utilization, or guaranteed latency.
Only requests sent through this process are counted. Keep one gateway process
while its reservation/session state is held in memory. Clients send conversation
history; affinity does not itself store chat messages.

The mock workers are independent FIFO HTTP processes and require no GPU,
model weights, vLLM or Fuzzball. For real worker addresses, authentication and
workflow records, see the [Fuzzball guide](../deploy/fuzzball/PRECISIONSPLIT.md).
A local source change does not update an already-running cluster gateway.

## Start the workers and gateway

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

For different ports, change the worker `--port` values and the corresponding
gateway `--bf16-url` / `--fp8-url` values together. Worker URLs are base
addresses without `/infer` or `/v1`; the forwarding code appends `/infer`.

## Run the routing demo

Run the demo with idle workers and the supplied default timing rules:

```sh
.venv-precisionsplit/bin/python -m precision_split.demo \
  --gateway-url http://127.0.0.1:9100 --output results/precisionsplit-demo.json
```

The demo verifies an idle tie, overflow to the other worker while one is
busy, and a follow-up that queues on its original worker. It prints request
counts and checks that all reservations are released.

Implementation: [routing.py](routing.py) handles accounting and affinity;
[forwarding.py](forwarding.py) handles HTTP; [vllm_gateway.py](vllm_gateway.py)
provides the real OpenAI-style API. The separate `gateway/` baseline also uses
least-outstanding routing with rotating ties.
