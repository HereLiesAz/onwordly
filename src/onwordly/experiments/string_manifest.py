from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from onwordly.tasks.string_manipulation import STRING_OPERATIONS


@dataclass(frozen=True, slots=True)
class StringExperimentManifest:
    model_name: str
    token_budget: int
    static_dataset_size: int
    evaluation_size: int
    composition_evaluation_size: int
    checkpoint_evaluation_size: int
    checkpoint_interval_tokens: int
    dataset_seed: int
    evaluation_seed: int
    training_seed: int
    learning_rate: float
    max_new_tokens: int
    lengths: tuple[int, ...]
    out_of_range_lengths: tuple[int, ...]
    operations: tuple[str, ...]
    alphabet: tuple[str, ...]
    variants_per_failure: int
    holdout_modulus: int
    # Tokens of shared answer-format warm-up, taken from the end of the static
    # training pool and trained identically before every regime; counted in
    # token_budget. 0 disables it.
    format_warmup_tokens: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "StringExperimentManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("lengths", "out_of_range_lengths", "operations", "alphabet"):
            payload[key] = tuple(payload[key])
        manifest = cls(**payload)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not 0 <= self.format_warmup_tokens < self.token_budget:
            raise ValueError("format_warmup_tokens must be in [0, token_budget)")
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if (
            self.static_dataset_size < 1
            or self.evaluation_size < 1
            or self.composition_evaluation_size < 1
        ):
            raise ValueError("dataset sizes must be positive")
        if not self.lengths or min(self.lengths) < 1:
            raise ValueError("lengths must contain positive integers")
        if not self.out_of_range_lengths or min(self.out_of_range_lengths) <= max(self.lengths):
            raise ValueError("out_of_range_lengths must be above the training range")
        if not self.operations or not set(self.operations).issubset(STRING_OPERATIONS):
            raise ValueError("operations contain an unsupported string value")
        if not self.alphabet or any(
            len(char) != 1 or not char.isalnum() or not char.isascii()
            for char in self.alphabet
        ):
            raise ValueError("alphabet must contain single ASCII alphanumeric characters")
        if self.checkpoint_evaluation_size < 1 or self.checkpoint_evaluation_size > self.evaluation_size:
            raise ValueError("invalid checkpoint_evaluation_size")
        if self.checkpoint_interval_tokens < 1:
            raise ValueError("checkpoint_interval_tokens must be positive")
        if self.variants_per_failure < 1:
            raise ValueError("variants_per_failure must be positive")
        if self.holdout_modulus < 2:
            raise ValueError("holdout_modulus must be at least 2")
