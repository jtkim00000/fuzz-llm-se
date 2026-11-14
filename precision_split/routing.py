"""Routing and reservation accounting, independent of HTTP forwarding."""

import asyncio
from dataclasses import asdict, dataclass
import json
import logging
import time
import uuid

from .timing import ServiceTime


logger = logging.getLogger("precision_split.routing")


@dataclass
class Reservation:
    request_id: str
    service_seconds: float
    started_at: float | None = None


@dataclass(frozen=True)
class Assignment:
    request_id: str
    session_id: str
    worker: str
    reason: str
    scores: dict

    def as_dict(self):
        return asdict(self)


class GreedyRouter:
    def __init__(self, timings: dict[str, ServiceTime], *, clock=time.monotonic):
        self.timings = timings
        self.clock = clock
        self._lock = asyncio.Lock()
        self._queues = {worker: [] for worker in sorted(timings)}
        self._sessions = {}

    def _remaining(self, worker, now):
        queue = self._queues[worker]
        if not queue:
            return 0.0
        head = queue[0]
        running = max(0.0, head.service_seconds - (now - head.started_at))
        return running + sum(item.service_seconds for item in queue[1:])

    async def reserve(self, session_id, input_length, expected_output_length):
        async with self._lock:
            now = self.clock()
            scores = {}
            for worker, timing in self.timings.items():
                remaining = self._remaining(worker, now)
                service = timing.estimate(input_length, expected_output_length)
                scores[worker] = {
                    "remaining_assigned_seconds": remaining,
                    "estimated_service_seconds": service,
                    "predicted_finish_seconds": remaining + service,
                    "assigned_requests_before": len(self._queues[worker]),
                }
            worker = self._sessions.get(session_id)
            reason = "session_affinity"
            if worker is None:
                worker = min(scores, key=lambda name: (scores[name]["predicted_finish_seconds"], name))
                reason = "minimum_predicted_finish_time"
            assignment = Assignment(uuid.uuid4().hex, session_id, worker, reason, scores)
            queue = self._queues[worker]
            queue.append(Reservation(
                assignment.request_id,
                scores[worker]["estimated_service_seconds"],
                started_at=now if not queue else None,
            ))
            self._sessions[session_id] = worker
            # Commit the decision and reservation before forwarding can yield.
            logger.info(json.dumps({
                "event": "assignment", **assignment.as_dict(),
                "assigned_requests_after": len(queue),
            }))
            return assignment

    async def release(self, assignment, outcome="finished"):
        async with self._lock:
            queue = self._queues[assignment.worker]
            index = next((i for i, item in enumerate(queue)
                          if item.request_id == assignment.request_id), None)
            if index is None:
                return False
            queue.pop(index)
            if index == 0 and queue:
                queue[0].started_at = self.clock()
            logger.info(json.dumps({
                "event": "release", "request_id": assignment.request_id,
                "worker": assignment.worker, "outcome": outcome,
                "assigned_requests_after": len(queue),
            }))
            return True

    async def snapshot(self):
        async with self._lock:
            now = self.clock()
            return {
                "workers": {worker: {
                    "assigned_requests": len(queue),
                    "estimated_remaining_seconds": self._remaining(worker, now),
                    "request_ids": [item.request_id for item in queue],
                    "timing": self.timings[worker].as_dict(),
                } for worker, queue in self._queues.items()},
                "sessions": dict(self._sessions),
            }
