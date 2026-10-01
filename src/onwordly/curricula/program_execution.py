from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

from onwordly.tasks.program_execution import (
    INSTRUCTION_NAMES,
    InstructionName,
    ProgramPartition,
    ProgramTask,
    generate_program_task,
    program_partition,
)


@dataclass(slots=True)
class ProgramBucketStats:
    attempts: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return 0.0 if self.attempts == 0 else self.correct / self.attempts


@dataclass(frozen=True, slots=True)
class ProgramBucket:
    length: int


class AdaptiveProgramCurriculum:
    def __init__(
        self,
        lengths: Sequence[int] = (3, 5, 8),
        *,
        operations: Sequence[InstructionName] = INSTRUCTION_NAMES,
        argument_min: int = -9,
        argument_max: int = 9,
        minimum_weight: float = 0.05,
        prior_attempts: int = 4,
        prior_correct: int = 2,
        partition: ProgramPartition | None = None,
        partition_modulus: int = 5,
    ) -> None:
        if not lengths or min(lengths) < 1:
            raise ValueError("lengths must contain positive values")
        if minimum_weight <= 0:
            raise ValueError("minimum_weight must be positive")
        if prior_attempts < 0 or prior_correct < 0 or prior_correct > prior_attempts:
            raise ValueError("invalid prior")
        self.operations = tuple(operations)
        self.argument_min = argument_min
        self.argument_max = argument_max
        self.minimum_weight = minimum_weight
        self.prior_attempts = prior_attempts
        self.prior_correct = prior_correct
        self.partition = partition
        self.partition_modulus = partition_modulus
        self.buckets = tuple(ProgramBucket(length) for length in lengths)
        self.stats = {bucket: ProgramBucketStats() for bucket in self.buckets}

    def _smoothed_accuracy(self, bucket: ProgramBucket) -> float:
        stats = self.stats[bucket]
        denominator = stats.attempts + self.prior_attempts
        if denominator == 0:
            return 0.0
        return (stats.correct + self.prior_correct) / denominator

    def weight(self, bucket: ProgramBucket) -> float:
        return max(self.minimum_weight, 1.0 - self._smoothed_accuracy(bucket))

    def choose_bucket(self, rng: Random) -> ProgramBucket:
        return rng.choices(
            self.buckets,
            weights=[self.weight(bucket) for bucket in self.buckets],
            k=1,
        )[0]

    def accepts(self, task: ProgramTask) -> bool:
        return (
            self.partition is None
            or program_partition(task, modulus=self.partition_modulus) == self.partition
        )

    def generate(self, rng: Random) -> ProgramTask:
        for _ in range(10_000):
            bucket = self.choose_bucket(rng)
            task = generate_program_task(
                rng,
                bucket.length,
                argument_min=self.argument_min,
                argument_max=self.argument_max,
                operations=self.operations,
            )
            if self.accepts(task):
                return task
        raise RuntimeError("could not generate a program task in the requested partition")

    def update(self, task: ProgramTask, correct: bool) -> None:
        bucket = ProgramBucket(task.length)
        if bucket not in self.stats:
            raise ValueError(f"task bucket is outside this curriculum: {bucket}")
        stats = self.stats[bucket]
        stats.attempts += 1
        stats.correct += int(correct)


class UniformProgramCurriculum(AdaptiveProgramCurriculum):
    def choose_bucket(self, rng: Random) -> ProgramBucket:
        return rng.choice(self.buckets)

    def update(self, task: ProgramTask, correct: bool) -> None:
        del task, correct
