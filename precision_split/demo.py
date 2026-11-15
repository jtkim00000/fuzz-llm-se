"""Exercise the three-process prototype using its public HTTP endpoints."""

import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid

import httpx


async def wait_for(get_state, predicate, description, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = await get_state()
        if predicate(state):
            return state
        await asyncio.sleep(0.01)
    raise RuntimeError(f"Timed out waiting for {description}")


async def run_demo(gateway_url):
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        async def state(url=gateway_url):
            response = await client.get(url.rstrip("/") + "/state")
            response.raise_for_status()
            return response.json()

        initial = await state()
        if any(w["assigned_requests"] for w in initial["workers"].values()):
            raise RuntimeError("Run the demo with idle workers and no other traffic")
        for mode, url in initial["worker_urls"].items():
            info = await state(url)
            assert info["mode"] == mode, "Worker URL points to the wrong mode"
            assert info["timing"] == initial["workers"][mode]["timing"], "Timing files differ"
            assert not info["running_request_id"] and info["queued_requests"] == 0, "Worker is busy"

        prefix = "demo-" + uuid.uuid4().hex[:10]

        async def infer(session, input_length=10, expected_output_length=10):
            response = await client.post(gateway_url.rstrip("/") + "/infer", json={
                "session_id": session, "input_length": input_length,
                "expected_output_length": expected_output_length,
            })
            response.raise_for_status()
            return response.json()

        idle_session = prefix + "-idle"
        idle = await infer(idle_session)
        preferred = idle["routing"]["worker"]
        other = next(w for w in initial["workers"] if w != preferred)
        scores = idle["routing"]["scores"]
        assert all(score["assigned_requests_before"] == 0 for score in scores.values())

        busy_session = idle_session
        busy_task = asyncio.create_task(infer(busy_session, 1000, 500))
        followup_task = None
        try:
            assigned = await wait_for(state, lambda s: s["workers"][preferred]["assigned_requests"] == 1, "busy assignment")
            assert assigned["sessions"][busy_session] == preferred, "Use the supplied default timing rules for this demo"
            worker_url = initial["worker_urls"][preferred]
            await wait_for(lambda: state(worker_url), lambda s: s["running_request_id"] is not None, "worker start")

            overflow = await infer(prefix + "-overflow")
            assert overflow["routing"]["worker"] == other
            assert overflow["routing"]["reason"] == "least_outstanding_requests"
            scores = overflow["routing"]["scores"]
            assert scores[other]["assigned_requests_before"] < scores[preferred]["assigned_requests_before"]

            followup_task = asyncio.create_task(infer(idle_session))
            queued = await wait_for(state, lambda s: s["workers"][preferred]["assigned_requests"] == 2,
                                    "follow-up reservation alongside running work")
            await wait_for(lambda: state(worker_url), lambda s: s["queued_requests"] == 1, "FIFO queue")
            busy, followup = await asyncio.gather(busy_task, followup_task)
            assert followup["routing"]["worker"] == preferred
            assert followup["routing"]["reason"] == "session_affinity"
            scores = followup["routing"]["scores"]
            assert scores[preferred]["assigned_requests_before"] > scores[other]["assigned_requests_before"]
            assert scores[other]["assigned_requests_before"] == 0
            assert busy["result"]["sequence"] < followup["result"]["sequence"]
            assert followup["result"]["queue_wait_seconds"] > 0
            final = await state()
            assert all(w["assigned_requests"] == 0 for w in final["workers"].values())
            return {"passed": True, "idle": idle, "busy_seed": busy, "overflow": overflow,
                    "followup": followup, "queued_snapshot": queued, "final_state": final}
        finally:
            tasks = [task for task in (busy_task, followup_task) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:9100")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(run_demo(args.gateway_url))
    for label in ("idle", "overflow", "followup"):
        decision = report[label]["routing"]
        scores = {mode: round(score["assigned_requests_before"], 4)
                  for mode, score in decision["scores"].items()}
        print(f"{label}: scores={scores} -> {decision['worker']} ({decision['reason']})")
    print("PASS: FIFO follow-up queued behind busy work; all reservations released")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
