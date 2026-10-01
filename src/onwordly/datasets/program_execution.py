from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Sequence

from onwordly.tasks.program_execution import (
    INSTRUCTION_NAMES,
    InstructionName,
    ProgramPartition,
    ProgramTask,
    generate_program_task,
    program_contains_transition,
    program_partition,
)


def build_static_program_dataset(
    *,
    seed: int,
    size: int,
    lengths: Sequence[int] = (3, 5, 8),
    operations: Sequence[InstructionName] = INSTRUCTION_NAMES,
    argument_min: int = -9,
    argument_max: int = 9,
    partition: ProgramPartition | None = None,
    partition_modulus: int = 5,
    forbidden_transitions: Sequence[tuple[InstructionName, InstructionName]] = (),
    shuffle: bool = True,
) -> tuple[ProgramTask, ...]:
    if size < 1:
        raise ValueError("size must be positive")
    if not lengths or min(lengths) < 1:
        raise ValueError("lengths must contain positive values")
    rng = Random(seed)
    tasks: list[ProgramTask] = []
    for index in range(size):
        length = lengths[index % len(lengths)]
        for _ in range(10_000):
            task = generate_program_task(
                rng,
                length,
                argument_min=argument_min,
                argument_max=argument_max,
                operations=operations,
            )
            if any(
                program_contains_transition(task, transition)
                for transition in forbidden_transitions
            ):
                continue
            if partition is None or program_partition(
                task,
                modulus=partition_modulus,
            ) == partition:
                tasks.append(task)
                break
        else:
            raise RuntimeError("could not generate a program task in the requested partition")
    if shuffle:
        rng.shuffle(tasks)
    return tuple(tasks)


def build_program_transition_dataset(
    *,
    seed: int,
    size: int,
    transition: tuple[InstructionName, InstructionName],
    lengths: Sequence[int] = (3, 5, 8),
    operations: Sequence[InstructionName] = INSTRUCTION_NAMES,
    argument_min: int = -9,
    argument_max: int = 9,
    partition: ProgramPartition | None = "eval",
    partition_modulus: int = 5,
    shuffle: bool = True,
) -> tuple[ProgramTask, ...]:
    if size < 1:
        raise ValueError("size must be positive")
    first, second = transition
    if first not in operations or second not in operations:
        raise ValueError("transition instructions must be available operations")
    if not lengths or max(lengths) < 2:
        raise ValueError("transition evaluation requires programs of length at least 2")

    rng = Random(seed)
    tasks: list[ProgramTask] = []
    for index in range(size):
        length = lengths[index % len(lengths)]
        if length < 2:
            continue
        for _ in range(10_000):
            task = generate_program_task(
                rng,
                length,
                argument_min=argument_min,
                argument_max=argument_max,
                operations=operations,
            )
            if not program_contains_transition(task, transition):
                continue
            if partition is None or program_partition(
                task,
                modulus=partition_modulus,
            ) == partition:
                tasks.append(task)
                break
        else:
            raise RuntimeError(
                "could not generate a program task with the required transition"
            )
    if len(tasks) != size:
        raise RuntimeError("could not fill transition evaluation dataset")
    if shuffle:
        rng.shuffle(tasks)
    return tuple(tasks)


def write_program_jsonl(tasks: Sequence[ProgramTask], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            payload = {
                "prompt": task.prompt,
                "answer": task.answer,
                "length": task.length,
                "instructions": [asdict(instruction) for instruction in task.instructions],
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return destination
