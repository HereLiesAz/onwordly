from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

from onwordly.tasks.string_manipulation import (
    STRING_OPERATIONS,
    ComposedStringTask,
    StringOperation,
    StringPartition,
    StringTask,
    composed_string_partition,
    generate_string_task,
    make_composed_string_task,
    string_partition,
)


def build_static_string_dataset(
    *,
    seed: int,
    size: int,
    operations: Iterable[StringOperation] = STRING_OPERATIONS,
    lengths: Iterable[int] = (4, 8, 12),
    alphabet: Sequence[str] = tuple("ABCDE12345"),
    partition: StringPartition | None = None,
    partition_modulus: int = 5,
    shuffle: bool = True,
) -> tuple[StringTask, ...]:
    if size < 1:
        raise ValueError("size must be positive")
    buckets = tuple(
        (operation, length)
        for operation in operations
        for length in lengths
    )
    if not buckets:
        raise ValueError("at least one string bucket is required")
    rng = Random(seed)
    tasks: list[StringTask] = []
    for index in range(size):
        operation, length = buckets[index % len(buckets)]
        for _ in range(10_000):
            task = generate_string_task(
                rng,
                operation,
                length,
                alphabet=alphabet,
            )
            if partition is None or string_partition(
                task,
                modulus=partition_modulus,
            ) == partition:
                tasks.append(task)
                break
        else:
            raise RuntimeError("could not generate a string task in the requested partition")
    if shuffle:
        rng.shuffle(tasks)
    return tuple(tasks)


def build_string_composition_dataset(
    *,
    seed: int,
    size: int,
    operations: Iterable[StringOperation] = STRING_OPERATIONS,
    lengths: Iterable[int] = (4, 8, 12),
    alphabet: Sequence[str] = tuple("ABCDE12345"),
    partition: StringPartition | None = "eval",
    partition_modulus: int = 5,
) -> tuple[ComposedStringTask, ...]:
    if size < 1:
        raise ValueError("size must be positive")
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
    normalized_alphabet = tuple(char.upper() for char in alphabet)
    tasks: list[ComposedStringTask] = []
    for index in range(size):
        first, second, length = pairs[index % len(pairs)]
        for _ in range(10_000):
            text = "".join(rng.choice(normalized_alphabet) for _ in range(length))
            task = make_composed_string_task(text, first, second)
            if partition is None or composed_string_partition(
                task,
                modulus=partition_modulus,
            ) == partition:
                tasks.append(task)
                break
        else:
            raise RuntimeError(
                "could not generate a composed string task in the requested partition"
            )
    rng.shuffle(tasks)
    return tuple(tasks)


def write_string_jsonl(
    tasks: Sequence[StringTask | ComposedStringTask],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
    return destination
