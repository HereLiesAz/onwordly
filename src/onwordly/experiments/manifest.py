from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from onwordly.tasks.arithmetic import PROMPT_STYLES


@dataclass(frozen=True, slots=True)
class ArithmeticExperimentManifest:
    model_name: str
    token_budget: int
    static_dataset_size: int
    evaluation_size: int
    generalization_size: int
    checkpoint_evaluation_size: int
    checkpoint_interval_tokens: int
    dataset_seed: int
    evaluation_seed: int
    training_seed: int
    learning_rate: float
    max_new_tokens: int
    digit_levels: tuple[int, ...]
    out_of_range_digit_levels: tuple[int, ...]
    operations: tuple[str, ...]
    withheld_prompt_styles: tuple[str, ...]
    variants_per_failure: int
    holdout_modulus: int
    # Optional LoRA config (r, alpha, dropout, target_modules). None = full fine-tuning.
    lora: dict[str, object] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ArithmeticExperimentManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for key in (
            "digit_levels",
            "out_of_range_digit_levels",
            "operations",
            "withheld_prompt_styles",
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
        if self.generalization_size < 1 or self.checkpoint_evaluation_size < 1:
            raise ValueError("evaluation subset sizes must be positive")
        if self.checkpoint_evaluation_size > self.evaluation_size:
            raise ValueError("checkpoint_evaluation_size cannot exceed evaluation_size")
        if self.checkpoint_interval_tokens < 1:
            raise ValueError("checkpoint_interval_tokens must be positive")
        if not self.digit_levels or min(self.digit_levels) < 1:
            raise ValueError("digit_levels must contain positive integers")
        if (
            not self.out_of_range_digit_levels
            or min(self.out_of_range_digit_levels) <= max(self.digit_levels)
        ):
            raise ValueError("out_of_range_digit_levels must be above the training range")
        allowed_operations = {"add", "subtract", "multiply"}
        if not self.operations or not set(self.operations).issubset(allowed_operations):
            raise ValueError("operations contain an unsupported value")
        if (
            not self.withheld_prompt_styles
            or "canonical" in self.withheld_prompt_styles
            or not set(self.withheld_prompt_styles).issubset(PROMPT_STYLES)
        ):
            raise ValueError("withheld_prompt_styles must be supported non-canonical styles")
        if self.variants_per_failure < 1:
            raise ValueError("variants_per_failure must be positive")
        if self.holdout_modulus < 2:
            raise ValueError("holdout_modulus must be at least 2")
        if self.lora is not None:
            missing = {"r", "alpha", "target_modules"} - set(self.lora)
            if missing:
                raise ValueError(f"lora is missing keys: {sorted(missing)}")
            if int(self.lora["r"]) < 1 or not self.lora["target_modules"]:
                raise ValueError("lora needs a positive r and at least one target module")
