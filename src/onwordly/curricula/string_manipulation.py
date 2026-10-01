from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, Sequence

from onwordly.tasks.string_manipulation import (
    STRING_OPERATIONS,
    StringOperation,
    StringPartition,
    StringTask,
    generate_string_task,
    string_partition,
)


@dataclass(slots=True)
class StringBucketStats:
    attempts: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return 0.0 if self.attempts == 0 else self.correct / self.attempts


@dataclass(frozen=True, slots=True)
class StringBucket:
    operation: StringOperation
    length: int


class AdaptiveStringCurriculum:
    def __init__(
        self,
        operations: Iterable[StringOperation] = STRING_OPERATIONS,
        lengths: Iterable[int] = (4, 8, 12),
        *,
        alphabet: Sequence[str] = tuple("ABCDE12345"),
        minimum_weight: float = 0.05,
        prior_attempts: int = 4,
        prior_correct: int = 2,
        partition: StringPartition | None = None,
        partition_modulus: int = 5,
    ) -> None:
        if minimum_weight <= 0:
            raise ValueError("minimum_weight must be positive")
        if prior_attempts < 0 or prior_correct < 0 or prior_correct > prior_attempts:
            raise ValueError("invalid prior")
        if partition_modulus < 2:
            raise ValueError("partition_modulus must be at least 2")
        self.minimum_weight = minimum_weight
        self.prior_attempts = prior_attempts
        self.prior_correct = prior_correct
        self.alphabet = tuple(alphabet)
        self.partition = partition
        self.partition_modulus = partition_modulus
        self.buckets = tuple(
            StringBucket(operation, length)
            for operation in operations
            for length in lengths
        )
        if not self.buckets or any(bucket.length < 1 for bucket in self.buckets):
            raise ValueError("string curriculum requires positive-length buckets")
        self.stats = {bucket: StringBucketStats() for bucket in self.buckets}

    def _smoothed_accuracy(self, bucket: StringBucket) -> float:
        stats = self.stats[bucket]
        denominator = stats.attempts + self.prior_attempts
        if denominator == 0:
            return 0.0
        return (stats.correct + self.prior_correct) / denominator

    def weight(self, bucket: StringBucket) -> float:
        return max(self.minimum_weight, 1.0 - self._smoothed_accuracy(bucket))

    def choose_bucket(self, rng: Random) -> StringBucket:
        return rng.choices(
            self.buckets,
            weights=[self.weight(bucket) for bucket in self.buckets],
            k=1,
        )[0]

    def accepts(self, task: StringTask) -> bool:
        return (
            self.partition is None
            or string_partition(task, modulus=self.partition_modulus) == self.partition
        )

    def generate(self, rng: Random) -> StringTask:
        for _ in range(10_000):
            bucket = self.choose_bucket(rng)
            task = generate_string_task(
                rng,
                bucket.operation,
                bucket.length,
                alphabet=self.alphabet,
            )
            if self.accepts(task):
                return task
        raise RuntimeError("could not generate a task in the string curriculum partition")

    def update(self, task: StringTask, correct: bool) -> None:
        bucket = StringBucket(task.operation, task.length)
        if bucket not in self.stats:
            raise ValueError(f"task bucket is outside this curriculum: {bucket}")
        stats = self.stats[bucket]
        stats.attempts += 1
        stats.correct += int(correct)


class UniformStringCurriculum(AdaptiveStringCurriculum):
    def choose_bucket(self, rng: Random) -> StringBucket:
        return rng.choice(self.buckets)

    def update(self, task: StringTask, correct: bool) -> None:
        del task, correct
