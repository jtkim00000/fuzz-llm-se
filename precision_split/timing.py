"""Shared service-time rules for worker delays and gateway predictions."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


DEFAULT_TIMINGS = Path(__file__).with_name("timings.json")


@dataclass(frozen=True)
class ServiceTime:
    base_seconds: float
    input_seconds_per_token: float
    output_seconds_per_token: float

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in asdict(self).values()):
            raise ValueError("Timing coefficients must be finite and nonnegative")

    def estimate(self, input_length: int, expected_output_length: int) -> float:
        if input_length < 0 or expected_output_length < 0:
            raise ValueError("Lengths must be nonnegative")
        return (
            self.base_seconds
            + self.input_seconds_per_token * input_length
            + self.output_seconds_per_token * expected_output_length
        )

    def as_dict(self):
        return asdict(self)


def load_timings(path=DEFAULT_TIMINGS):
    data = json.loads(Path(path).read_text())
    if set(data) != {"bf16", "fp8"}:
        raise ValueError("Timing configuration must define exactly bf16 and fp8")
    return {mode: ServiceTime(**values) for mode, values in data.items()}
