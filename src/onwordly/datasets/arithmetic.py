from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

from onwordly.tasks.arithmetic import ArithmeticTask, Operation, generate_arithmetic_task


def build_static_arithmetic_dataset(
    *,
    seed: int,
    size: int,
    operations: Iterable[Operation] = ("add", "subtract", "multiply"),
    digit_levels: Iterable[int] = (1, 2, 3),
    shuffle: bool = True,
) -> tuple[ArithmeticTask, ...]:
    """Build a deterministic, approximately balanced frozen arithmetic dataset."""
    if size < 1:
        raise ValueError("size must be at least 1")

    buckets = tuple((operation, digits) for operation in operations for digits in digit_levels)
    if not buckets:
        raise ValueError("at least one operation/difficulty bucket is required")
    if any(digits < 1 for _, digits in buckets):
        raise ValueError("digit levels must be at least 1")

    rng = Random(seed)
    tasks = [
        generate_arithmetic_task(rng, *buckets[index % len(buckets)])
        for index in range(size)
    ]
    if shuffle:
        rng.shuffle(tasks)
    return tuple(tasks)


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
