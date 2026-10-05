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

    def sample(self, prompt: str, n: int, temperature: float) -> list[str]:
        """``n`` independent sampled completions (on-policy training only)."""
        ...

    def train_weighted(self, prompt: str, completion: str, weight: float) -> TrainStepMetrics:
        """One update on ``weight`` x the completion's mean negative log-likelihood.

        ``tokens`` must equal ``count_training_tokens(prompt, completion)``.
        """
        ...
