"""Save a short-lived Fuzzball endpoint token in a private local file."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expiration", default="2h")
    args = parser.parse_args()
    executable = shutil.which("fuzzball") or str(Path.home() / "bin/fuzzball")
    result = subprocess.run([executable, "workflow", "endpoints", "generate-token",
                             args.endpoint_id, "--expiration", args.expiration, "-o", "json"],
                            capture_output=True, text=True, check=True, timeout=30)
    data = json.loads(result.stdout)
    token = data if isinstance(data, str) else next((data[k] for k in
        ("token", "accessToken", "access_token", "bearerToken") if isinstance(data.get(k), str)), None)
    if not token or any(c.isspace() for c in token):
        raise RuntimeError("Endpoint token response has an unexpected format")
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(token + "\n")
    print(json.dumps({"token_file": str(args.output), "permissions": "0600", "expiration": args.expiration}))


if __name__ == "__main__":
    main()
