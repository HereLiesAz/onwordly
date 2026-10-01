from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArithmeticExperimentManifest:
    model_name: str
    token_budget: int
    static_dataset_size: int
    evaluation_size: int
    dataset_seed: int
    evaluation_seed: int
    training_seed: int
    learning_rate: float
    max_new_tokens: int
    digit_levels: tuple[int, ...]
    operations: tuple[str, ...]
    variants_per_failure: int

    @classmethod
    def from_json(cls, path: str | Path) -> "ArithmeticExperimentManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["digit_levels"] = tuple(payload["digit_levels"])
        payload["operations"] = tuple(payload["operations"])
        manifest = cls(**payload)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if self.static_dataset_size < 1 or self.evaluation_size < 1:
            raise ValueError("dataset sizes must be positive")
        if not self.digit_levels or min(self.digit_levels) < 1:
            raise ValueError("digit_levels must contain positive integers")
        allowed = {"add", "subtract", "multiply"}
        if not self.operations or not set(self.operations).issubset(allowed):
            raise ValueError("operations contain an unsupported value")
        if self.variants_per_failure < 1:
            raise ValueError("variants_per_failure must be positive")
