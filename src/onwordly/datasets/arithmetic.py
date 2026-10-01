from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

from onwordly.tasks.arithmetic import (
    ArithmeticPartition,
    ArithmeticTask,
    Operation,
    PromptStyle,
    arithmetic_task_identity,
    generate_arithmetic_task,
    task_in_partition,
)


def build_static_arithmetic_dataset(
    *,
    seed: int,
    size: int,
    operations: Iterable[Operation] = ("add", "subtract", "multiply"),
    digit_levels: Iterable[int] = (1, 2, 3),
    prompt_styles: Iterable[PromptStyle] = ("canonical",),
    partition: ArithmeticPartition | None = None,
    partition_modulus: int = 5,
    shuffle: bool = True,
) -> tuple[ArithmeticTask, ...]:
    """Build a deterministic, approximately balanced frozen arithmetic dataset."""
    if size < 1:
        raise ValueError("size must be at least 1")

    buckets = tuple(
        (operation, digits, prompt_style)
        for operation in operations
        for digits in digit_levels
        for prompt_style in prompt_styles
    )
    if not buckets:
        raise ValueError("at least one operation/difficulty/style bucket is required")
    if any(digits < 1 for _, digits, _ in buckets):
        raise ValueError("digit levels must be at least 1")

    rng = Random(seed)
    tasks: list[ArithmeticTask] = []
    max_attempts_per_task = 10_000

    for index in range(size):
        operation, digits, prompt_style = buckets[index % len(buckets)]
        for _ in range(max_attempts_per_task):
            task = generate_arithmetic_task(
                rng,
                operation,
                digits,
                prompt_style=prompt_style,
            )
            if task_in_partition(task, partition, modulus=partition_modulus):
                tasks.append(task)
                break
        else:
            raise RuntimeError(
                "could not generate a task in the requested partition; "
                "check the bucket and partition settings"
            )

    if shuffle:
        rng.shuffle(tasks)
    return tuple(tasks)


def arithmetic_dataset_stats(tasks: Sequence[ArithmeticTask]) -> dict[str, object]:
    """Summarize logical and presentation-level duplication in a dataset."""
    if not tasks:
        raise ValueError("dataset cannot be empty")

    logical = {arithmetic_task_identity(task) for task in tasks}
    presented = {
        arithmetic_task_identity(task, include_prompt_style=True)
        for task in tasks
    }
    by_bucket: dict[str, list[ArithmeticTask]] = {}
    for task in tasks:
        by_bucket.setdefault(f"{task.operation}:{task.digits}", []).append(task)

    def summarize(group: Sequence[ArithmeticTask]) -> dict[str, int | float]:
        identities = {arithmetic_task_identity(task) for task in group}
        duplicates = len(group) - len(identities)
        return {
            "examples": len(group),
            "unique_logical_tasks": len(identities),
            "duplicate_examples": duplicates,
            "duplicate_rate": duplicates / len(group),
        }

    return {
        "examples": len(tasks),
        "unique_logical_tasks": len(logical),
        "unique_presented_tasks": len(presented),
        "duplicate_logical_examples": len(tasks) - len(logical),
        "duplicate_logical_rate": (len(tasks) - len(logical)) / len(tasks),
        "by_bucket": {
            key: summarize(group)
            for key, group in sorted(by_bucket.items())
        },
    }


def write_arithmetic_jsonl(tasks: Sequence[ArithmeticTask], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
    return destination


def read_arithmetic_jsonl(path: str | Path) -> tuple[ArithmeticTask, ...]:
    source = Path(path)
    tasks: list[ArithmeticTask] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                tasks.append(ArithmeticTask(**payload))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"invalid arithmetic JSONL at line {line_number}") from exc
    if not tasks:
        raise ValueError("dataset is empty")
    return tuple(tasks)
