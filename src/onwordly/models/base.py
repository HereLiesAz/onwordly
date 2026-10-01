from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class TrainStepMetrics:
    loss: float
    tokens: int


@runtime_checkable
class ModelAdapter(Protocol):
    """Small interface required by the training and evaluation harnesses."""

    def generate(self, prompt: str) -> str:
        ...

    def count_training_tokens(self, prompt: str, target: str) -> int:
        ...

    def train_example(self, prompt: str, target: str) -> TrainStepMetrics:
        ...
