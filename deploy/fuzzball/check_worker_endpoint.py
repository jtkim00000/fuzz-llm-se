"""Check a private vLLM endpoint from the client machine, without logging credentials."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.endpoint.startswith("https://"):
        parser.error("Use the HTTPS Fuzzball endpoint")
    token = args.token_file.read_text().strip()
    if not token or any(character.isspace() for character in token):
        parser.error("The token file must contain only the bearer token")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "endpoint": args.endpoint,
              "model": args.model, "checks": {}}
    # No redirect following: never forward authentication to a different host.
    with httpx.Client(base_url=args.endpoint.rstrip("/") + "/", timeout=180,
                      headers={"Authorization": "Bearer " + token}) as client:
        response = client.get("health")
        response.raise_for_status()
        result["checks"]["health"] = response.status_code
        response = client.get("v1/models")
        response.raise_for_status()
        models = response.json()
        assert args.model in [model["id"] for model in models["data"]], models
        result["checks"]["models"] = models
        metrics = client.get("metrics")
        result["checks"]["metrics_status"] = metrics.status_code
        if metrics.is_success:
            (args.output / "metrics-before.prom").write_text(metrics.text)
        payload = {"model": args.model, "messages": [{"role": "user", "content": "Reply with the single word READY."}],
                   "max_tokens": 128, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        started = time.monotonic()
        response = client.post("v1/chat/completions", json=payload)
        response.raise_for_status()
        completion = response.json()
        assert completion["choices"][0]["message"].get("content"), completion
        result["checks"]["completion"] = {"seconds": time.monotonic() - started, "response": completion}
        started = time.monotonic()
        first_content = None
        chunks, done = [], False
        with client.stream("POST", "v1/chat/completions", json={**payload, "stream": True,
                           "stream_options": {"include_usage": True}}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                chunks.append(chunk)
                if first_content is None and any(choice.get("delta", {}).get("content") for choice in chunk.get("choices", [])):
                    first_content = time.monotonic() - started
        assert done and first_content is not None, "Streaming did not return content followed by [DONE]"
        result["checks"]["streaming"] = {"seconds": time.monotonic() - started,
            "first_content_seconds": first_content, "done": done, "chunks": chunks}
        metrics = client.get("metrics")
        if metrics.is_success:
            (args.output / "metrics-after.prom").write_text(metrics.text)
    (args.output / "endpoint-checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"health": "passed", "models": "passed", "completion": "passed", "streaming": "passed",
                      "metrics_status": result["checks"]["metrics_status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
