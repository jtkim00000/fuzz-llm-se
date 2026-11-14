#!/usr/bin/env python3
"""Bundle current checked-in tools into a standalone Fuzzball v4 preparation job."""
import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = sorted(p.name for p in ROOT.iterdir() if p.is_file() and
               (p.suffix in ['.py', '.json'] or p.name == 'requirements-training.txt'))

def render():
    payload = base64.b64encode(json.dumps({
        name: base64.b64encode((ROOT / name).read_bytes()).decode()
        for name in FILES
    }).encode()).decode()
    script = '''#!/bin/sh
set -eu
RUN_DIR=$(mktemp -d /results/baseline-prep-XXXXXXXX)
export RUN_DIR
python - <<'PYCODE'
import os, json, base64
from pathlib import Path
root = Path(os.environ['RUN_DIR']) / 'tools'
root.mkdir()
files = json.loads(base64.b64decode(PAYLOAD))
for name, data in files.items():
    (root / name).write_bytes(base64.b64decode(data))
PYCODE
python "$RUN_DIR/tools/grade.py"
python "$RUN_DIR/tools/build_suite.py" --out "$RUN_DIR/inputs/reactome-v1-preview"
python "$RUN_DIR/tools/selftest.py" "$RUN_DIR/inputs/reactome-v1-preview"
printf 'PREPARED_ROOT=%s\\n' "$RUN_DIR"
'''.replace('PAYLOAD', repr(payload))
    return '''version: v4
volumes:
  datasets:
    name: datasets
  results:
    name: profiling-results
jobs:
  prepare-baseline:
    image:
      uri: docker://python:3.11
    mounts:
      /datasets: datasets
      /results: results
    resource:
      cpu:
        cores: 2
      memory:
        size: 4GB
    script: |
''' + ''.join('      ' + line + '\n' for line in script.splitlines())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'generated/prepare-baseline.yaml')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render())
    print(args.out)
