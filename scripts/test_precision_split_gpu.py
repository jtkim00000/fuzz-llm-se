"""Run four short real-model requests through a deployed PrecisionSplit gateway."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import ssl
import time
import uuid

import httpx


async def check(url, token=None, ca_file=None):
    headers = {"Authorization": "Bearer " + token} if token else {}
    context = ssl.create_default_context()
    if ca_file:
        context.load_verify_locations(cafile=str(ca_file))
    async with httpx.AsyncClient(base_url=url.rstrip("/") + "/", headers=headers,
                                 timeout=120, trust_env=False, verify=context) as client:
        health = await client.get("health")
        health.raise_for_status()
        assert all(health.json()["workers"].values()), health.json()
        response = await client.get("v1/models")
        response.raise_for_status()
        model = response.json()["data"][0]["id"]
        prefix = "gpu-check-" + uuid.uuid4().hex[:10]
        sessions = [prefix + "-a", prefix + "-b"]

        async def submit(session, prompt, limit):
            started = time.monotonic()
            response = await client.post("v1/chat/completions", headers={"X-Session-ID": session}, json={
                "model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": limit, "temperature": 0,
            })
            response.raise_for_status()
            body = response.json()
            assert body["choices"][0]["message"].get("content"), body
            return {"session_id": session, "worker": response.headers["x-precisionsplit-worker"],
                    "request_id": response.headers["x-precisionsplit-request-id"],
                    "seconds": time.monotonic() - started, "response": body}

        concurrent = await asyncio.gather(*(submit(session,
            "List the integers from 1 through 200, separated by spaces. Continue until the output limit.", 64)
            for session in sessions))
        assert len({r["worker"] for r in concurrent}) == 2, "Concurrent sessions did not reach both workers"
        followup = await submit(sessions[0], "Reply with the single word READY.", 8)
        assert followup["worker"] == concurrent[0]["worker"], "Session affinity changed"
        started = time.monotonic()
        content, usage, first_content, done = "", None, None, False
        async with client.stream("POST", "v1/chat/completions", headers={"X-Session-ID": prefix + "-stream"}, json={
            "model": model, "messages": [{"role": "user", "content": "Reply with READY."}],
            "max_tokens": 8, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
        }) as response:
            response.raise_for_status()
            streaming_worker = response.headers["x-precisionsplit-worker"]
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    continue
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    piece = choice.get("delta", {}).get("content") or ""
                    if piece and first_content is None:
                        first_content = time.monotonic() - started
                    content += piece
        assert done and content, "Stream did not complete"
        for _ in range(30):
            state_response = await client.get("state")
            state_response.raise_for_status()
            state = state_response.json()
            if all(w["assigned_requests"] == 0 for w in state["workers"].values()):
                break
            await asyncio.sleep(0.1)
        assert all(w["assigned_requests"] == 0 for w in state["workers"].values())
        return {"passed": True, "checked_at": datetime.now(timezone.utc).isoformat(),
                "gateway_url": url, "model": model, "health": health.json(),
                "concurrent": concurrent, "followup": followup,
                "streaming": {"worker": streaming_worker, "done": done, "content": content,
                              "usage": usage, "first_content_seconds": first_content},
                "final_state": state, "scope": "Functional routing smoke test, not a performance or accuracy benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--ca-file", type=Path, help="Approved cluster CA; TLS verification remains enabled")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    token = args.token_file.read_text().strip() if args.token_file else None
    report = asyncio.run(check(args.url, token, args.ca_file))
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
