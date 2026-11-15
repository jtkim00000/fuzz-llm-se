import argparse

import uvicorn

from .app import WorkerConfig, create_app


def main():
    parser = argparse.ArgumentParser(description="Start a localhost worker facade for router tests")
    parser.add_argument("--worker", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--simulate-fp8", action="store_true")
    args = parser.parse_args()
    if args.worker == "fp8" and not args.simulate_fp8:
        parser.error("Metal cannot serve FP8 KV caches. Pass --simulate-fp8 for an explicitly labeled test stand-in.")
    if args.worker == "bf16" and args.simulate_fp8:
        parser.error("--simulate-fp8 is only valid for the fp8 test worker")
    config = WorkerConfig(
        worker_id=args.worker,
        requested_kv_cache_dtype="bfloat16" if args.worker == "bf16" else "fp8",
        backend_url=args.backend_url,
        simulated=args.simulate_fp8,
    )
    uvicorn.run(create_app(config), host="127.0.0.1", port=8001 if args.worker == "bf16" else 8002)


if __name__ == "__main__":
    main()
