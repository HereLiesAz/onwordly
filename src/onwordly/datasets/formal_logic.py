from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Sequence

from onwordly.tasks.formal_logic import (
    LogicPartition,
    LogicTask,
    LogicKind,
    generate_logic_task,
    logic_contains_operator_composition,
    logic_partition,
)


def build_static_logic_dataset(
    *,
    seed: int,
    size: int,
    depths: Sequence[int] = (1, 2, 3),
    variables: Sequence[str] = ("A", "B", "C", "D"),
    partition: LogicPartition | None = None,
    partition_modulus: int = 5,
    forbidden_compositions: Sequence[tuple[LogicKind, LogicKind]] = (),
    shuffle: bool = True,
) -> tuple[LogicTask, ...]:
    if size < 1:
        raise ValueError("size must be positive")
    if not depths or min(depths) < 0:
        raise ValueError("depths must contain non-negative values")
    rng = Random(seed)
    tasks: list[LogicTask] = []
    for index in range(size):
        depth = eligible_depths[index % len(eligible_depths)]
        for _ in range(10_000):
            task = generate_logic_task(rng, depth, variables=variables)
            if any(
                logic_contains_operator_composition(task.expression, composition)
                for composition in forbidden_compositions
            ):
                continue
            if partition is None or logic_partition(
                task,
                modulus=partition_modulus,
            ) == partition:
                tasks.append(task)
                break
        else:
            raise RuntimeError("could not generate a logic task in the requested partition")
    if shuffle:
        rng.shuffle(tasks)
    return tuple(tasks)


def build_logic_composition_dataset(
    *,
    seed: int,
    size: int,
    composition: tuple[LogicKind, LogicKind],
    depths: Sequence[int] = (2, 3),
    variables: Sequence[str] = ("A", "B", "C", "D"),
    partition: LogicPartition | None = "eval",
    partition_modulus: int = 5,
    shuffle: bool = True,
) -> tuple[LogicTask, ...]:
    if size < 1:
        raise ValueError("size must be positive")
    eligible_depths = tuple(depth for depth in depths if depth >= 2)
    if not eligible_depths:
        raise ValueError("composition evaluation requires depth at least 2")
    parent_kind, child_kind = composition
    if parent_kind == "var" or child_kind == "var":
        raise ValueError("operator composition cannot contain var")

    rng = Random(seed)
    tasks: list[LogicTask] = []
    for index in range(size):
        depth = depths[index % len(depths)]
        for _ in range(20_000):
            task = generate_logic_task(rng, depth, variables=variables)
            if not logic_contains_operator_composition(task.expression, composition):
                continue
            if partition is None or logic_partition(
                task,
                modulus=partition_modulus,
            ) == partition:
                tasks.append(task)
                break
        else:
            raise RuntimeError(
                "could not generate a logic task with the requested operator composition"
            )
    if shuffle:
        rng.shuffle(tasks)
    return tuple(tasks)


def write_logic_jsonl(tasks: Sequence[LogicTask], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            payload = {
                "prompt": task.prompt,
                "answer": task.answer,
                "expression": task.expression.render(),
                "assignment": list(task.assignment),
                "depth": task.depth,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return destination
