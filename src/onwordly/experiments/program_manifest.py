from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from onwordly.tasks.program_execution import INSTRUCTION_NAMES


@dataclass(frozen=True, slots=True)
class ProgramExperimentManifest:
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
    lengths: tuple[int, ...]
    out_of_range_lengths: tuple[int, ...]
    operations: tuple[str, ...]
    withheld_transition: tuple[str, str]
    argument_min: int
    argument_max: int
    variants_per_failure: int
    holdout_modulus: int

    @classmethod
    def from_json(cls, path: str | Path) -> "ProgramExperimentManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in (
            "lengths",
            "out_of_range_lengths",
            "operations",
            "withheld_transition",
        ):
            payload[key] = tuple(payload[key])
        manifest = cls(**payload)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if self.static_dataset_size < 1 or self.evaluation_size < 1:
            raise ValueError("dataset sizes must be positive")
        if not self.lengths or min(self.lengths) < 1:
            raise ValueError("lengths must contain positive integers")
        if not self.out_of_range_lengths or min(self.out_of_range_lengths) <= max(self.lengths):
            raise ValueError("out_of_range_lengths must be above training lengths")
        if not self.operations or not set(self.operations).issubset(INSTRUCTION_NAMES):
            raise ValueError("operations contain an unsupported instruction")
        if (
            len(self.withheld_transition) != 2
            or not set(self.withheld_transition).issubset(self.operations)
        ):
            raise ValueError(
                "withheld_transition must contain two available instructions"
            )
        if self.argument_min > self.argument_max:
            raise ValueError("argument_min cannot exceed argument_max")
        if self.checkpoint_evaluation_size < 1 or self.checkpoint_evaluation_size > self.evaluation_size:
            raise ValueError("invalid checkpoint_evaluation_size")
        if self.checkpoint_interval_tokens < 1:
            raise ValueError("checkpoint_interval_tokens must be positive")
        if self.variants_per_failure < 1:
            raise ValueError("variants_per_failure must be positive")
        if self.holdout_modulus < 2:
            raise ValueError("holdout_modulus must be at least 2")
