from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, Sequence

from onwordly.tasks.symbolic import (
    SYMBOLIC_OPERATIONS,
    SymbolicOperation,
    SymbolicPartition,
    SymbolicTask,
    generate_symbolic_task,
    symbolic_task_in_partition,
)


@dataclass(slots=True)
class SymbolicBucketStats:
    attempts: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return 0.0 if self.attempts == 0 else self.correct / self.attempts


@dataclass(frozen=True, slots=True)
class SymbolicBucket:
    operation: SymbolicOperation
    length: int


class AdaptiveSymbolicCurriculum:
    def __init__(
        self,
        operations: Iterable[SymbolicOperation] = SYMBOLIC_OPERATIONS,
        lengths: Iterable[int] = (4, 8, 12),
        *,
        alphabet: Sequence[str] = ("A", "B", "C", "D", "E"),
        minimum_weight: float = 0.05,
        prior_attempts: int = 4,
        prior_correct: int = 2,
        partition: SymbolicPartition | None = None,
        partition_modulus: int = 5,
    ) -> None:
        if minimum_weight <= 0:
            raise ValueError("minimum_weight must be positive")
        if prior_attempts < 0 or prior_correct < 0 or prior_correct > prior_attempts:
            raise ValueError("invalid prior")
        self.minimum_weight = minimum_weight
        self.prior_attempts = prior_attempts
        self.prior_correct = prior_correct
        self.alphabet = tuple(alphabet)
        self.partition = partition
        self.partition_modulus = partition_modulus
        self.buckets = tuple(
            SymbolicBucket(operation, length)
            for operation in operations
            for length in lengths
        )
        if not self.buckets or any(bucket.length < 1 for bucket in self.buckets):
            raise ValueError("symbolic curriculum requires positive-length buckets")
        self.stats = {bucket: SymbolicBucketStats() for bucket in self.buckets}

    def _smoothed_accuracy(self, bucket: SymbolicBucket) -> float:
        stats = self.stats[bucket]
        denominator = stats.attempts + self.prior_attempts
        if denominator == 0:
            return 0.0
        return (stats.correct + self.prior_correct) / denominator

    def weight(self, bucket: SymbolicBucket) -> float:
        return max(self.minimum_weight, 1.0 - self._smoothed_accuracy(bucket))

    def choose_bucket(self, rng: Random) -> SymbolicBucket:
        return rng.choices(
            self.buckets,
            weights=[self.weight(bucket) for bucket in self.buckets],
            k=1,
        )[0]

    def accepts(self, task: SymbolicTask) -> bool:
        return symbolic_task_in_partition(
            task,
            self.partition,
            modulus=self.partition_modulus,
        )

    def generate(self, rng: Random) -> SymbolicTask:
        for _ in range(10_000):
            bucket = self.choose_bucket(rng)
            task = generate_symbolic_task(
                rng,
                bucket.operation,
                bucket.length,
                alphabet=self.alphabet,
            )
            if self.accepts(task):
                return task
        raise RuntimeError("could not generate a task in the symbolic curriculum partition")

    def update(self, task: SymbolicTask, correct: bool) -> None:
        bucket = SymbolicBucket(task.operation, task.length)
        if bucket not in self.stats:
            raise ValueError(f"task bucket is outside this curriculum: {bucket}")
        stats = self.stats[bucket]
        stats.attempts += 1
        stats.correct += int(correct)


class UniformSymbolicCurriculum(AdaptiveSymbolicCurriculum):
    def choose_bucket(self, rng: Random) -> SymbolicBucket:
        return rng.choice(self.buckets)

    def update(self, task: SymbolicTask, correct: bool) -> None:
        del task, correct
