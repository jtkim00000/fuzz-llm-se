import asyncio
import math
import unittest

from fastapi.testclient import TestClient
import httpx

from precision_split.apps import create_gateway, create_worker
from precision_split.routing import GreedyRouter
from precision_split.timing import ServiceTime, load_timings


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_simultaneous_arrivals_see_prior_reservations_and_rotate_ties(self):
        router = GreedyRouter({w: ServiceTime(1, 0, 0) for w in ("fp8", "bf16")}, clock=lambda: 0)
        assignments = await asyncio.gather(*(router.reserve(f"session-{i}", 0, 0) for i in range(6)))
        self.assertEqual([a.worker for a in assignments], ["bf16", "fp8"] * 3)
        state = await router.snapshot()
        self.assertEqual([w["assigned_requests"] for w in state["workers"].values()], [3, 3])

    async def test_running_work_decays_but_queued_work_remains_fully_reserved(self):
        now = [0.0]
        router = GreedyRouter({"bf16": ServiceTime(2, 0, 0), "fp8": ServiceTime(10, 0, 0)}, clock=lambda: now[0])
        first = await router.reserve("a", 0, 0)
        second = await router.reserve("a", 0, 0)
        now[0] = 1.25
        state = await router.snapshot()
        self.assertEqual(state["workers"]["bf16"]["estimated_remaining_seconds"], 2.75)
        await router.release(first)
        self.assertEqual((await router.snapshot())["workers"]["bf16"]["estimated_remaining_seconds"], 2)
        now[0] = 1.75
        self.assertEqual((await router.snapshot())["workers"]["bf16"]["estimated_remaining_seconds"], 1.5)
        await router.release(second)
        self.assertEqual((await router.snapshot())["workers"]["bf16"]["assigned_requests"], 0)
        self.assertFalse(await router.release(second))

    async def test_busy_worker_sends_new_session_elsewhere_but_keeps_followup(self):
        router = GreedyRouter(load_timings(), clock=lambda: 0)
        first = await router.reserve("original", 10, 10)
        await router.release(first)
        busy = await router.reserve("original", 1000, 500)
        overflow = await router.reserve("new", 10, 10)
        await router.release(overflow)
        followup = await router.reserve("original", 10, 10)
        self.assertEqual((first.worker, busy.worker, overflow.worker, followup.worker), ("bf16", "bf16", "fp8", "bf16"))
        self.assertEqual(followup.reason, "session_affinity")
        self.assertGreater(followup.scores["bf16"]["assigned_requests_before"], followup.scores["fp8"]["assigned_requests_before"])
        self.assertEqual((await router.snapshot())["workers"]["bf16"]["assigned_requests"], 2)

    async def test_counts_override_timings_even_after_estimate_expires(self):
        now = [0.0]
        router = GreedyRouter({"bf16": ServiceTime(0.01, 0, 0),
                               "fp8": ServiceTime(100, 0, 0)}, clock=lambda: now[0])
        first = await router.reserve("long-running", 0, 0)
        now[0] = 500
        second = await router.reserve("new", 0, 0)
        self.assertEqual((first.worker, second.worker), ("bf16", "fp8"))
        self.assertEqual(second.reason, "least_outstanding_requests")
        await router.release(first)
        await router.release(second)
        third = await router.reserve("idle-a", 1000, 1)
        await router.release(third)
        fourth = await router.reserve("idle-b", 1, 1000)
        self.assertEqual((third.worker, fourth.worker), ("bf16", "fp8"))

    async def test_failed_queued_reservation_does_not_reset_running_progress(self):
        now = [0.0]
        router = GreedyRouter({"bf16": ServiceTime(2, 0, 0), "fp8": ServiceTime(100, 0, 0)}, clock=lambda: now[0])
        await router.reserve("a", 0, 0)
        queued = await router.reserve("a", 0, 0)
        now[0] = 0.5
        await router.release(queued, "failed")
        self.assertEqual((await router.snapshot())["workers"]["bf16"]["estimated_remaining_seconds"], 1.5)

    async def test_cancelled_http_forwarding_releases_reservation(self):
        started = asyncio.Event()

        async def upstream(request):
            started.set()
            await asyncio.Event().wait()

        app = create_gateway(load_timings(), {"bf16": "http://bf16", "fp8": "http://fp8"}, transport=httpx.MockTransport(upstream))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
                task = asyncio.create_task(client.post("/infer", json={"input_length": 10, "expected_output_length": 10}))
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(all(w["assigned_requests"] == 0 for w in (await app.state.router.snapshot())["workers"].values()))

    async def test_worker_fifo_allows_only_one_running_request(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def controlled_sleep(seconds):
            calls.append(seconds)
            entered.set()
            await release.wait()

        app = create_worker("bf16", load_timings(), sleep=controlled_sleep)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
                async def send(request_id):
                    return await client.post("/infer", json={"request_id": request_id, "input_length": 10, "expected_output_length": 10})

                first = asyncio.create_task(send("first"))
                await asyncio.wait_for(entered.wait(), 1)
                second = asyncio.create_task(send("second"))
                for _ in range(100):
                    if app.state.worker.queue.qsize() == 1:
                        break
                    await asyncio.sleep(0)
                self.assertEqual(app.state.worker.snapshot()["queued_requests"], 1)
                self.assertEqual(app.state.worker.snapshot()["running_request_id"], "first")
                self.assertEqual(len(calls), 1)
                release.set()
                results = await asyncio.gather(first, second)
                self.assertEqual([r.json()["sequence"] for r in results], [1, 2])
                self.assertEqual(len(calls), 2)


class GatewayTests(unittest.TestCase):
    def test_upstream_failures_release_load_and_keep_session_assignment(self):
        for failure in ("connect", "status", "invalid_json", "timeout"):
            with self.subTest(failure=failure):
                def upstream(request):
                    if failure == "connect":
                        raise httpx.ConnectError("stopped", request=request)
                    if failure == "timeout":
                        raise httpx.ReadTimeout("timeout", request=request)
                    if failure == "invalid_json":
                        return httpx.Response(200, text="not JSON")
                    return httpx.Response(500)

                app = create_gateway(load_timings(), {"bf16": "http://bf16", "fp8": "http://fp8"}, transport=httpx.MockTransport(upstream))
                with TestClient(app) as client:
                    response = client.post("/infer", json={"session_id": "a", "input_length": 10, "expected_output_length": 10})
                    self.assertEqual(response.status_code, 504 if failure == "timeout" else 502)
                    state = client.get("/state").json()
                    self.assertTrue(all(w["assigned_requests"] == 0 for w in state["workers"].values()))
                    self.assertEqual(state["sessions"]["a"], "bf16")

    def test_invalid_request_does_not_reserve_work(self):
        app = create_gateway(load_timings(), {"bf16": "http://bf16", "fp8": "http://fp8"})
        with TestClient(app) as client:
            self.assertEqual(client.post("/infer", json={"input_length": -1, "expected_output_length": 1}).status_code, 422)
            self.assertEqual(client.get("/state").json()["sessions"], {})

    def test_nonfinite_or_negative_timing_is_rejected(self):
        for value in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                ServiceTime(value, 0, 0)


if __name__ == "__main__":
    unittest.main()
