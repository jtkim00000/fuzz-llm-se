"""Fetch the pinned tokenizer and public ARM64 image metadata; never model weights."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CHECKPOINT = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
REVISION = "6efb4a2a1c1fa277ce7b3df7a1416255011b1c99"
IMAGE_TAG = "25.12.post1-py3"


def fetch(url, headers=None):
    with urlopen(Request(url, headers=headers or {}), timeout=30) as response:
        return response.read(), dict(response.headers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"checkpoint": CHECKPOINT, "revision": REVISION, "files": {}, "errors": {}}
    tokenizer = args.output / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    def download(name):
        try:
            data, _ = fetch(f"https://huggingface.co/{CHECKPOINT}/resolve/{REVISION}/{name}?download=true")
            (tokenizer / name).write_bytes(data)
            return name, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, None
        except Exception as error:
            return name, None, type(error).__name__

    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "chat_template.jinja", "config.json")
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        for name, details, error in pool.map(download, names):
            if error:
                result["errors"][name] = error
            else:
                result["files"][name] = details
    # Anonymous pull token stays in memory and is never serialized or logged.
    try:
        query = urlencode({"service": "nvcr.io", "scope": "repository:nvidia/vllm:pull"})
        data, _ = fetch("https://nvcr.io/proxy_auth?" + query)
        token_data = json.loads(data)
        token = token_data.get("token") or token_data["access_token"]
        headers = {"Authorization": "Bearer " + token,
                   "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"}
        data, response_headers = fetch(f"https://nvcr.io/v2/nvidia/vllm/manifests/{IMAGE_TAG}", headers)
        index = json.loads(data)
        digest = response_headers.get("Docker-Content-Digest") or response_headers.get("docker-content-digest")
        arm64 = [item for item in index.get("manifests", []) if item.get("platform", {}).get("architecture") == "arm64"]
        result["image"] = {"tag": "nvcr.io/nvidia/vllm:" + IMAGE_TAG,
                           "index_digest": digest, "arm64_manifests": arm64,
                           "source": "public registry lookup; runtime image identity must be checked separately"}
        (args.output / "image-index.json").write_bytes(data)
    except Exception as error:
        result["image"] = {"error_type": type(error).__name__, "digest": None}
    (args.output / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tokenizer_path": str(tokenizer.resolve()), "files": len(result["files"]),
                      "errors": result["errors"], "image": result["image"]}, indent=2))


if __name__ == "__main__":
    main()
