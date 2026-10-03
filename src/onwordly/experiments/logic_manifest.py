from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LogicExperimentManifest:
    model_name: str
    token_budget: int
    static_dataset_size: int
    evaluation_size: int
    checkpoint_evaluation_size: int
    checkpoint_interval_tokens: int
    dataset_seed: int
    evaluation_seed: int
    training_seed: int
    learning_rate: float
    max_new_tokens: int
    depths: tuple[int, ...]
    out_of_range_depths: tuple[int, ...]
    variables: tuple[str, ...]
    withheld_composition: tuple[str, str]
    composition_evaluation_size: int
    variants_per_failure: int
    holdout_modulus: int
    # Tokens of shared answer-format warm-up, taken from the end of the static
    # training pool and trained identically before every regime; counted in
    # token_budget. 0 disables it.
    format_warmup_tokens: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "LogicExperimentManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("depths", "out_of_range_depths", "variables", "withheld_composition"):
            payload[key] = tuple(payload[key])
        manifest = cls(**payload)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not 0 <= self.format_warmup_tokens < self.token_budget:
            raise ValueError("format_warmup_tokens must be in [0, token_budget)")
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if self.static_dataset_size < 1 or self.evaluation_size < 1:
            raise ValueError("dataset sizes must be positive")
        if not self.depths or min(self.depths) < 0:
            raise ValueError("depths must contain non-negative integers")
        if not self.out_of_range_depths or min(self.out_of_range_depths) <= max(self.depths):
            raise ValueError("out_of_range_depths must exceed training depths")
        if not self.variables or any(
            len(name) != 1 or not name.isalpha() or not name.isascii()
            for name in self.variables
        ):
            raise ValueError("variables must contain single ASCII letters")
        if len(self.withheld_composition) != 2 or any(
            kind not in {"not", "and", "or", "xor"}
            for kind in self.withheld_composition
        ):
            raise ValueError("withheld_composition must contain two logic operators")
        if self.composition_evaluation_size < 1:
            raise ValueError("composition_evaluation_size must be positive")
        if self.checkpoint_evaluation_size < 1 or self.checkpoint_evaluation_size > self.evaluation_size:
            raise ValueError("invalid checkpoint_evaluation_size")
        if self.checkpoint_interval_tokens < 1:
            raise ValueError("checkpoint_interval_tokens must be positive")
        if self.variants_per_failure < 1:
            raise ValueError("variants_per_failure must be positive")
        if self.holdout_modulus < 2:
            raise ValueError("holdout_modulus must be at least 2")
