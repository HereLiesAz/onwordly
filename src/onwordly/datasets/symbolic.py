from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

from onwordly.tasks.symbolic import (
    SYMBOLIC_OPERATIONS,
    ComposedSymbolicTask,
    SymbolicOperation,
    SymbolicPartition,
    SymbolicTask,
    composed_symbolic_partition,
    generate_symbolic_task,
    make_composed_symbolic_task,
    symbolic_task_in_partition,
)


def build_static_symbolic_dataset(
    *,
    seed: int,
    size: int,
    operations: Iterable[SymbolicOperation] = SYMBOLIC_OPERATIONS,
    lengths: Iterable[int] = (4, 8, 12),
    alphabet: Sequence[str] = ("A", "B", "C", "D", "E"),
    partition: SymbolicPartition | None = None,
    partition_modulus: int = 5,
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

    if partition_modulus < 2:
        raise ValueError("partition_modulus must be at least 2")

    rng = Random(seed)
    tasks: list[SymbolicTask] = []
    for index in range(size):
        operation, length = buckets[index % len(buckets)]
        for _ in range(10_000):
            task = generate_symbolic_task(
                rng,
                operation,
                length,
                alphabet=alphabet,
            )
            if symbolic_task_in_partition(
                task,
                partition,
                modulus=partition_modulus,
            ):
                tasks.append(task)
                break
        else:
            raise RuntimeError(
                "could not generate a symbolic task in the requested partition"
            )
    if shuffle:
        rng.shuffle(tasks)
    return tuple(tasks)


def build_symbolic_composition_dataset(
    *,
    seed: int,
    size: int,
    operations: Iterable[SymbolicOperation] = SYMBOLIC_OPERATIONS,
    lengths: Iterable[int] = (4, 8, 12),
    alphabet: Sequence[str] = ("A", "B", "C", "D", "E"),
    partition: SymbolicPartition | None = "eval",
    partition_modulus: int = 5,
) -> tuple[ComposedSymbolicTask, ...]:
    if size < 1:
        raise ValueError("size must be at least 1")
    pairs = tuple(
        (first, second, length)
        for first in operations
        for second in operations
        if first != second
        for length in lengths
    )
    if not pairs:
        raise ValueError("composition requires at least two distinct operations")

    rng = Random(seed)
    tasks: list[ComposedSymbolicTask] = []
    normalized_alphabet = tuple(symbol.upper() for symbol in alphabet)
    for index in range(size):
        first, second, length = pairs[index % len(pairs)]
        for _ in range(10_000):
            symbols = "".join(rng.choice(normalized_alphabet) for _ in range(length))
            task = make_composed_symbolic_task(symbols, first, second)
            if partition is None or composed_symbolic_partition(
                task,
                modulus=partition_modulus,
            ) == partition:
                tasks.append(task)
                break
        else:
            raise RuntimeError(
                "could not generate a composed symbolic task in the requested partition"
            )
    rng.shuffle(tasks)
    return tuple(tasks)


def write_symbolic_jsonl(tasks: Sequence[SymbolicTask | ComposedSymbolicTask], path: str | Path) -> Path:
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
