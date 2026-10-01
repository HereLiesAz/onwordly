from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

from onwordly.tasks.symbolic import (
    SYMBOLIC_OPERATIONS,
    SymbolicOperation,
    SymbolicTask,
    generate_symbolic_task,
)


def build_static_symbolic_dataset(
    *,
    seed: int,
    size: int,
    operations: Iterable[SymbolicOperation] = SYMBOLIC_OPERATIONS,
    lengths: Iterable[int] = (4, 8, 12),
    alphabet: Sequence[str] = ("A", "B", "C", "D", "E"),
    shuffle: bool = True,
) -> tuple[SymbolicTask, ...]:
    if size < 1:
        raise ValueError("size must be at least 1")
    buckets = tuple(
        (operation, length)
        for operation in operations
        for length in lengths
    )
    if not buckets:
        raise ValueError("at least one symbolic bucket is required")
    if any(length < 1 for _, length in buckets):
        raise ValueError("lengths must be positive")

    rng = Random(seed)
    tasks = [
        generate_symbolic_task(
            rng,
            operation,
            length,
            alphabet=alphabet,
        )
        for operation, length in (
            buckets[index % len(buckets)]
            for index in range(size)
        )
    ]
    if shuffle:
        rng.shuffle(tasks)
    return tuple(tasks)


def write_symbolic_jsonl(tasks: Sequence[SymbolicTask], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
    return destination


def read_symbolic_jsonl(path: str | Path) -> tuple[SymbolicTask, ...]:
    source = Path(path)
    tasks: list[SymbolicTask] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                tasks.append(SymbolicTask(**json.loads(stripped)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"invalid symbolic JSONL at line {line_number}"
                ) from exc
    if not tasks:
        raise ValueError("dataset is empty")
    return tuple(tasks)
