"""Start, inspect, or stop only this project's two local HTTP worker facades."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".local-inference/workers"
PORTS = {"bf16": 8001, "fp8": 8002}
HTTP = build_opener(ProxyHandler({}))


def get_json(url):
    with HTTP.open(url, timeout=2) as response:
        return json.load(response)


def info(worker):
    try:
        data = get_json(f"http://127.0.0.1:{PORTS[worker]}/worker-info")
        if data.get("application") == "precision-worker-test-facade" and data.get("worker_id") == worker:
            return data
    except (URLError, TimeoutError, ValueError):
        pass
    return None


def start():
    try:
        models = get_json("http://127.0.0.1:8000/v1/models")
        if "llama-3.1-8b-local" not in [item["id"] for item in models["data"]]:
            raise ValueError("Unexpected model on port 8000")
    except (URLError, TimeoutError, ValueError, KeyError) as exc:
        raise SystemExit("Start the existing Llama backend first: bash .local-inference/serve.sh") from exc

    for worker, port in PORTS.items():
        if info(worker):
            continue
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise SystemExit(f"Port {port} is occupied by another application; leaving it untouched.") from exc

    STATE.mkdir(parents=True, exist_ok=True)
    for worker in PORTS:
        if info(worker):
            continue
        command = [sys.executable, "-m", "local_workers", "--worker", worker]
        if worker == "fp8":
            command.append("--simulate-fp8")
        with (STATE / f"{worker}.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (STATE / f"{worker}.pid").write_text(str(process.pid) + "\n")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if info(worker):
                break
            if process.poll() is not None:
                raise SystemExit(f"Worker {worker} exited. See {STATE / f'{worker}.log'}")
            time.sleep(0.2)
        else:
            process.terminate()
            raise SystemExit(f"Worker {worker} did not become ready. See its log.")
    return status()


def status():
    all_ready = True
    for worker, port in PORTS.items():
        data = info(worker)
        try:
            ready = bool(data) and get_json(f"http://127.0.0.1:{port}/health")["ready"]
        except (URLError, TimeoutError, ValueError, KeyError):
            ready = False
        all_ready = all_ready and ready
        print(json.dumps({"worker": worker, "api_base": f"http://127.0.0.1:{port}/v1", "ready": ready, "precision_simulated": data["precision_simulated"] if data else None}))
    return 0 if all_ready else 1


def stop():
    for worker in PORTS:
        data = info(worker)
        pid_file = STATE / f"{worker}.pid"
        if not data:
            print(f"{worker}: no matching test worker is listening")
            continue
        if not pid_file.exists() or int(pid_file.read_text()) != data["process_id"]:
            print(f"{worker}: PID ownership could not be verified; leaving it running")
            continue
        os.kill(data["process_id"], signal.SIGTERM)
        pid_file.unlink()
        print(f"{worker}: stopped")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    args = parser.parse_args()
    sys.exit({"start": start, "status": status, "stop": stop}[args.action]())
