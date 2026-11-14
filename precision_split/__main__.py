import argparse
import json
import logging
from pathlib import Path

import uvicorn

from .apps import create_gateway, create_worker
from .timing import DEFAULT_TIMINGS, load_timings


def main():
    parser = argparse.ArgumentParser(description="PrecisionSplit local mock serving prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("worker", "gateway"):
        command = commands.add_parser(name)
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, required=True)
        command.add_argument("--timings", default=str(DEFAULT_TIMINGS))
        if name == "worker":
            command.add_argument("--mode", choices=("bf16", "fp8"), required=True)
        else:
            command.add_argument("--bf16-url", default="http://127.0.0.1:9101")
            command.add_argument("--fp8-url", default="http://127.0.0.1:9102")
            command.add_argument("--worker-timeout", type=float, default=30.0)
    real = commands.add_parser("vllm-gateway", help="Route real OpenAI-compatible vLLM requests")
    real.add_argument("--config", required=True)
    real.add_argument("--host", default="127.0.0.1")
    real.add_argument("--port", type=int, default=19000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.command == "vllm-gateway":
        from .vllm_gateway import create_vllm_gateway
        app = create_vllm_gateway(json.loads(Path(args.config).read_text()))
    elif args.command == "worker":
        timings = load_timings(args.timings)
        app = create_worker(args.mode, timings)
    else:
        timings = load_timings(args.timings)
        app = create_gateway(timings, {"bf16": args.bf16_url, "fp8": args.fp8_url}, timeout=args.worker_timeout)
    # One process owns each in-memory queue / reservation ledger.
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
