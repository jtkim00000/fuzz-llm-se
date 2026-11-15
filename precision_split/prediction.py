"""Portable repository forest inference; scores are E2E milliseconds under load."""

import hashlib
import math
from pathlib import Path
import time

from gateway.policy import load_model
from profiling.features import feature_row, predict


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "gateway/models/vast-full-v1/model.json"


class CompletionPredictor:
    def __init__(self, workers, model_path=None):
        self.workers = workers
        path = Path(model_path) if model_path else DEFAULT_MODEL
        self.artifact = load_model(path)
        self.version = hashlib.sha256(path.read_bytes()).hexdigest()

    def score(self, request, snapshots, outstanding):
        now = time.time_ns()
        scores = {}
        for name, count in outstanding.items():
            info = self.workers[name]
            manifest = dict(info.get("manifest", {}))
            dtype = manifest.get("effective_kv_dtype", info.get("effective_kv_dtype"))
            if dtype in ("bf16", "bfloat16", "torch.bfloat16"):
                dtype = "bfloat16"
            elif isinstance(dtype, str) and (dtype.startswith("fp8") or "float8" in dtype):
                dtype = "fp8"
            else:
                raise ValueError(f"Unknown or unsupported KV precision for {name}")
            manifest["effective_kv_dtype"] = dtype
            snapshot = snapshots.get(name, {"sample_completed_ns": now, "state": {}})
            features = feature_row(request, snapshot, {"manifest": manifest}, count, now)
            value = predict(features, self.artifact, "e2e_ms")
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid E2E prediction")
            outside = [key for key, (lo, hi) in self.artifact["bounds"].items()
                       if (features.get(key) is None and not self.artifact.get("missing_allowed", {}).get(key, False))
                       or (features.get(key) is not None and not lo <= features[key] <= hi)]
            scores[name] = {
                "predicted_completion_ms": value,
                "model_version": self.version,
                "features": features,
                "outside_training_features": outside,
                "training_configuration_match": info.get("configuration_hash") in self.artifact["configuration_hashes"],
                "telemetry_error": snapshot.get("error"),
            }
        return scores


def request_features(payload, tokens, budget):
    """Only information available before dispatch; never response-derived values."""
    return {
        "input_tokens_estimate": tokens,
        "output_limit": budget,
        "estimated_total_tokens": tokens + budget,
        "previous_turns": sum(m.get("role") == "assistant" for m in payload["messages"]),
        "streaming": bool(payload.get("stream")),
        "tool_call_mode": bool(payload.get("tools")),
        "structured_output": bool(payload.get("response_format") or payload.get("structured_outputs")),
        "thinking_enabled": (payload.get("chat_template_kwargs") or {}).get("enable_thinking"),
    }
