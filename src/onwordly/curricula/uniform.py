from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable

from onwordly.tasks.arithmetic import (
    ArithmeticPartition,
    ArithmeticTask,
    Operation,
    generate_arithmetic_task,
    task_in_partition,
)


@dataclass(frozen=True, slots=True)
class UniformBucket:
    operation: Operation
    digits: int


class UniformArithmeticCurriculum:
    """Generate online arithmetic tasks uniformly without competence adaptation."""

    def __init__(
        self,
        operations: Iterable[Operation] = ("add", "subtract", "multiply"),
        digit_levels: Iterable[int] = (1, 2, 3),
        *,
        partition: ArithmeticPartition | None = None,
        partition_modulus: int = 5,
    ) -> None:
        if partition_modulus < 2:
            raise ValueError("partition_modulus must be at least 2")
        self.partition = partition
        self.partition_modulus = partition_modulus
        self.buckets = tuple(
            UniformBucket(operation=operation, digits=digits)
            for operation in operations
            for digits in digit_levels
        )
        if not self.buckets:
            raise ValueError("at least one bucket is required")
        if any(bucket.digits < 1 for bucket in self.buckets):
            raise ValueError("digit levels must be at least 1")

    def accepts(self, task: ArithmeticTask) -> bool:
        return task_in_partition(task, self.partition, modulus=self.partition_modulus)

    def generate(self, rng: Random) -> ArithmeticTask:
        for _ in range(10_000):
            bucket = rng.choice(self.buckets)
            task = generate_arithmetic_task(rng, bucket.operation, bucket.digits)
            if self.accepts(task):
                return task
        raise RuntimeError("could not generate a task in the curriculum partition")

    def update(self, task: ArithmeticTask, correct: bool) -> None:
        del task, correct
