from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable

from onwordly.tasks.arithmetic import ArithmeticTask, Operation, generate_arithmetic_task


@dataclass(slots=True)
class BucketStats:
    attempts: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.correct / self.attempts


@dataclass(frozen=True, slots=True)
class Bucket:
    operation: Operation
    digits: int


class AdaptiveArithmeticCurriculum:
    """Sample more often from operation/difficulty buckets with lower accuracy."""

    def __init__(
        self,
        operations: Iterable[Operation] = ("add", "subtract", "multiply"),
        digit_levels: Iterable[int] = (1, 2, 3),
        *,
        minimum_weight: float = 0.05,
        prior_attempts: int = 4,
        prior_correct: int = 2,
    ) -> None:
        if minimum_weight <= 0:
            raise ValueError("minimum_weight must be positive")
        if prior_attempts < 0 or prior_correct < 0 or prior_correct > prior_attempts:
            raise ValueError("invalid prior")

        self.minimum_weight = minimum_weight
        self.prior_attempts = prior_attempts
        self.prior_correct = prior_correct
        self.buckets = tuple(
            Bucket(operation=operation, digits=digits)
            for operation in operations
            for digits in digit_levels
        )
        if not self.buckets:
            raise ValueError("at least one bucket is required")
        if any(bucket.digits < 1 for bucket in self.buckets):
            raise ValueError("digit levels must be at least 1")

        self.stats = {bucket: BucketStats() for bucket in self.buckets}

    def _smoothed_accuracy(self, bucket: Bucket) -> float:
        stats = self.stats[bucket]
        denominator = stats.attempts + self.prior_attempts
        if denominator == 0:
            return 0.0
        return (stats.correct + self.prior_correct) / denominator

    def weight(self, bucket: Bucket) -> float:
        weakness = 1.0 - self._smoothed_accuracy(bucket)
        return max(self.minimum_weight, weakness)

    def choose_bucket(self, rng: Random) -> Bucket:
        weights = [self.weight(bucket) for bucket in self.buckets]
        return rng.choices(self.buckets, weights=weights, k=1)[0]

    def generate(self, rng: Random) -> ArithmeticTask:
        bucket = self.choose_bucket(rng)
        return generate_arithmetic_task(rng, bucket.operation, bucket.digits)

    def update(self, task: ArithmeticTask, correct: bool) -> None:
        bucket = Bucket(task.operation, task.digits)
        if bucket not in self.stats:
            raise ValueError(f"task bucket is outside this curriculum: {bucket}")
        stats = self.stats[bucket]
        stats.attempts += 1
        stats.correct += int(correct)
