from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from random import Random
from typing import Literal, Sequence

InstructionName = Literal["SET", "ADD", "SUB", "MUL", "NEG"]
ProgramPartition = Literal["train", "eval"]
INSTRUCTION_NAMES: tuple[InstructionName, ...] = ("SET", "ADD", "SUB", "MUL", "NEG")


@dataclass(frozen=True, slots=True)
class Instruction:
    name: InstructionName
    argument: int | None = None

    def render(self) -> str:
        return self.name if self.argument is None else f"{self.name} {self.argument}"


@dataclass(frozen=True, slots=True)
class ProgramTask:
    prompt: str
    answer: int
    instructions: tuple[Instruction, ...]
    length: int

    @property
    def target_text(self) -> str:
        return str(self.answer)

    @property
    def bucket_key(self) -> str:
        return f"program:{self.length}"

    @property
    def program_text(self) -> str:
        return "; ".join(instruction.render() for instruction in self.instructions)


def execute_program(instructions: Sequence[Instruction]) -> int:
    if not instructions:
        raise ValueError("program cannot be empty")
    accumulator = 0
    for instruction in instructions:
        if instruction.name == "SET":
            if instruction.argument is None:
                raise ValueError("SET requires an argument")
            accumulator = instruction.argument
        elif instruction.name == "ADD":
            if instruction.argument is None:
                raise ValueError("ADD requires an argument")
            accumulator += instruction.argument
        elif instruction.name == "SUB":
            if instruction.argument is None:
                raise ValueError("SUB requires an argument")
            accumulator -= instruction.argument
        elif instruction.name == "MUL":
            if instruction.argument is None:
                raise ValueError("MUL requires an argument")
            accumulator *= instruction.argument
        elif instruction.name == "NEG":
            if instruction.argument is not None:
                raise ValueError("NEG takes no argument")
            accumulator = -accumulator
        else:
            raise ValueError(f"unsupported instruction: {instruction.name}")
    return accumulator


def make_program_task(instructions: Sequence[Instruction]) -> ProgramTask:
    normalized = tuple(instructions)
    answer = execute_program(normalized)
    program_text = "; ".join(instruction.render() for instruction in normalized)
    prompt = (
        "Execute this accumulator program from left to right. The accumulator starts at 0. "
        "SET replaces it, ADD/SUB/MUL apply the integer argument, and NEG changes its sign. "
        f"Program: {program_text}. Return only the final integer."
    )
    return ProgramTask(
        prompt=prompt,
        answer=answer,
        instructions=normalized,
        length=len(normalized),
    )


def generate_program_task(
    rng: Random,
    length: int,
    *,
    argument_min: int = -9,
    argument_max: int = 9,
    operations: Sequence[InstructionName] = INSTRUCTION_NAMES,
) -> ProgramTask:
    if length < 1:
        raise ValueError("length must be positive")
    if argument_min > argument_max:
        raise ValueError("argument_min cannot exceed argument_max")
    if not operations:
        raise ValueError("operations cannot be empty")

    instructions: list[Instruction] = []
    for index in range(length):
        available = tuple(operations)
        if index == 0 and "SET" in available:
            name = "SET"
        else:
            name = rng.choice(available)
        argument = None if name == "NEG" else rng.randint(argument_min, argument_max)
        instructions.append(Instruction(name=name, argument=argument))
    return make_program_task(instructions)


def program_partition(
    task: ProgramTask,
    *,
    modulus: int = 5,
) -> ProgramPartition:
    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    key = task.program_text.encode("utf-8")
    residue = int.from_bytes(blake2b(key, digest_size=8).digest(), "big") % modulus
    return "eval" if residue == 0 else "train"
