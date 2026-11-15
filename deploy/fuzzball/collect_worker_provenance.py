"""Collect non-secret startup provenance inside the requested serving image."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import torch
import vllm


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect():
    expected_gpu = os.environ.get("PROFILING_EXPECTED_GPU_UUID")
    visible_gpus = command(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
    if expected_gpu and (not visible_gpus or expected_gpu not in visible_gpus.splitlines()):
        raise RuntimeError("Requested SPARK03 GPU is not visible; refusing to start inference")
    model_dir = Path(os.environ["PROFILING_MODEL_PATH"])
    config = json.loads((model_dir / "config.json").read_text())
    properties = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    git_commit = command(["git", "-C", "/opt/vllm", "rev-parse", "HEAD"])
    if not git_commit:
        direct_url = importlib.metadata.distribution("vllm").read_text("direct_url.json")
        if direct_url:
            git_commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    data = {
        "model_checkpoint": os.environ["PROFILING_CHECKPOINT"],
        "model_revision": os.environ["PROFILING_MODEL_REVISION"],
        "model_path": str(model_dir),
        "engine_version": vllm.__version__,
        "engine_commit": git_commit,
        "engine_package_path": str(Path(vllm.__file__).resolve()),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "gpu": {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "multiprocessor_count": properties.multi_processor_count,
        } if properties else None,
        "gpu_query": command(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"]),
        "nvidia_smi": command(["nvidia-smi"]),
        "gpu_processes_before_model_start": command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]),
        "nvcc": command(["nvcc", "--version"]),
        "model_config": config,
        "selected_environment": {key: os.environ.get(key) for key in ["VLLM_USE_FLASHINFER_MOE_FP4", "VLLM_FLASHINFER_MOE_BACKEND", "CUDA_VERSION", "NVIDIA_BUILD_ID"]},
    }
    output = Path("/data/profiling/runtime-provenance.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print("PROFILING_PROVENANCE " + json.dumps(data), flush=True)


if __name__ == "__main__":
    collect()
