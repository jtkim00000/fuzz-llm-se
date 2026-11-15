"""Package this checkout into a CPU-only Fuzzball gateway workflow."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "deploy/fuzzball/precision-split-qwen38.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19038)
    args = parser.parse_args()
    source = {p.name: p.read_text() for p in (ROOT / "precision_split").iterdir()
              if p.suffix == ".py" or p.name == "timings.json"}
    config = json.loads(args.config.read_text())
    script = """#!/bin/sh
set -eu
python3 - <<'PY'
import json
from pathlib import Path
target = Path('/tmp/precision-split/precision_split')
target.mkdir(parents=True, exist_ok=True)
for name, content in json.loads(Path('/opt/precision-split-source.json').read_text()).items():
    (target / name).write_text(content)
PY
export PYTHONPATH=/tmp/precision-split
exec python3 -m precision_split vllm-gateway --config /opt/precision-split-config.json --host 0.0.0.0 --port PORT_NUMBER
""".replace("PORT_NUMBER", str(args.port))
    workflow = {
        "version": "v4",
        "files": {"source.json": json.dumps(source), "config.json": json.dumps(config)},
        "services": {"gateway": {
            # Reuse the already cached image's HTTP dependencies; no GPU is requested.
            "image": {"uri": "docker://vllm/vllm-openai:v0.28.0"},
            "persist": True,
            "resource": {"cpu": {"cores": 1}, "memory": {"size": "2GiB"}},
            "files": {"/opt/precision-split-source.json": "file://source.json",
                      "/opt/precision-split-config.json": "file://config.json"},
            "script": script,
            "network": {"ports": [{"name": "api", "port": args.port, "protocol": "tcp"}],
                        "endpoints": [{"name": "api", "port-name": "api", "type": "subdomain",
                                       "protocol": "http", "scope": "user"}]},
            "readiness-probe": {"http-get": {"path": "/health", "port": args.port, "scheme": "HTTP"},
                                "initial-delay-seconds": 5, "period-seconds": 5,
                                "timeout-seconds": 10, "failure-threshold": 120},
        }},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # JSON is a YAML subset accepted by Fuzzball's v4 parser.
    args.output.write_text(json.dumps(workflow, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
