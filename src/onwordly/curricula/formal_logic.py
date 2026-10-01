from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

from onwordly.tasks.formal_logic import (
    LogicPartition,
    LogicTask,
    generate_logic_task,
    logic_partition,
)


@dataclass(slots=True)
class LogicBucketStats:
    attempts: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return 0.0 if self.attempts == 0 else self.correct / self.attempts


@dataclass(frozen=True, slots=True)
class LogicBucket:
    depth: int


class AdaptiveLogicCurriculum:
    def __init__(
        self,
        depths: Sequence[int] = (1, 2, 3),
        *,
        variables: Sequence[str] = ("A", "B", "C", "D"),
        minimum_weight: float = 0.05,
        prior_attempts: int = 4,
        prior_correct: int = 2,
        partition: LogicPartition | None = None,
        partition_modulus: int = 5,
    ) -> None:
        if not depths or min(depths) < 0:
            raise ValueError("depths must contain non-negative values")
        if minimum_weight <= 0:
            raise ValueError("minimum_weight must be positive")
        if prior_attempts < 0 or prior_correct < 0 or prior_correct > prior_attempts:
            raise ValueError("invalid prior")
        if partition_modulus < 2:
            raise ValueError("partition_modulus must be at least 2")
        self.variables = tuple(name.upper() for name in variables)
        self.minimum_weight = minimum_weight
        self.prior_attempts = prior_attempts
        self.prior_correct = prior_correct
        self.partition = partition
        self.partition_modulus = partition_modulus
        self.buckets = tuple(LogicBucket(depth) for depth in depths)
        self.stats = {bucket: LogicBucketStats() for bucket in self.buckets}

    def _smoothed_accuracy(self, bucket: LogicBucket) -> float:
        stats = self.stats[bucket]
        denominator = stats.attempts + self.prior_attempts
        if denominator == 0:
            return 0.0
        return (stats.correct + self.prior_correct) / denominator

    def weight(self, bucket: LogicBucket) -> float:
        return max(self.minimum_weight, 1.0 - self._smoothed_accuracy(bucket))

    def choose_bucket(self, rng: Random) -> LogicBucket:
        return rng.choices(
            self.buckets,
            weights=[self.weight(bucket) for bucket in self.buckets],
            k=1,
        )[0]

    def accepts(self, task: LogicTask) -> bool:
        return (
            self.partition is None
            or logic_partition(task, modulus=self.partition_modulus) == self.partition
        )

    def generate(self, rng: Random) -> LogicTask:
        for _ in range(10_000):
            bucket = self.choose_bucket(rng)
            task = generate_logic_task(
                rng,
                bucket.depth,
                variables=self.variables,
            )
            if self.accepts(task):
                return task
        raise RuntimeError("could not generate a logic task in the requested partition")

    def update(self, task: LogicTask, correct: bool) -> None:
        bucket = LogicBucket(task.depth)
        if bucket not in self.stats:
            raise ValueError(f"task bucket is outside this curriculum: {bucket}")
        stats = self.stats[bucket]
        stats.attempts += 1
        stats.correct += int(correct)


class UniformLogicCurriculum(AdaptiveLogicCurriculum):
    def choose_bucket(self, rng: Random) -> LogicBucket:
        return rng.choice(self.buckets)

    def update(self, task: LogicTask, correct: bool) -> None:
        del task, correct
