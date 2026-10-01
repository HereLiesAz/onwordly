from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from random import Random
from typing import Literal, Sequence

SymbolicOperation = Literal["reverse", "sort", "dedupe_adjacent", "rotate_left"]
SymbolicPartition = Literal["train", "eval"]

SYMBOLIC_OPERATIONS: tuple[SymbolicOperation, ...] = (
    "reverse",
    "sort",
    "dedupe_adjacent",
    "rotate_left",
)

_RULE_TEXT: dict[SymbolicOperation, str] = {
    "reverse": "reverse the sequence",
    "sort": "sort the symbols in ascending alphabetical order",
    "dedupe_adjacent": "replace each run of repeated adjacent symbols with one symbol",
    "rotate_left": "move the first symbol to the end",
}


@dataclass(frozen=True, slots=True)
class SymbolicTask:
    prompt: str
    answer: str
    operation: SymbolicOperation
    length: int
    symbols: str

    @property
    def target_text(self) -> str:
        return self.answer

    @property
    def bucket_key(self) -> str:
        return f"{self.operation}:{self.length}"


def symbolic_partition(
    symbols: str,
    operation: SymbolicOperation,
    *,
    modulus: int = 5,
) -> SymbolicPartition:
    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    key = f"{operation}:{symbols.upper()}".encode("utf-8")
    residue = int.from_bytes(blake2b(key, digest_size=8).digest(), "big") % modulus
    return "eval" if residue == 0 else "train"


def symbolic_task_in_partition(
    task: SymbolicTask,
    partition: SymbolicPartition | None,
    *,
    modulus: int = 5,
) -> bool:
    if partition is None:
        return True
    return symbolic_partition(task.symbols, task.operation, modulus=modulus) == partition


def apply_symbolic_operation(symbols: str, operation: SymbolicOperation) -> str:
    if not symbols:
        raise ValueError("symbols cannot be empty")
    if operation == "reverse":
        return symbols[::-1]
    if operation == "sort":
        return "".join(sorted(symbols))
    if operation == "dedupe_adjacent":
        output = [symbols[0]]
        for symbol in symbols[1:]:
            if symbol != output[-1]:
                output.append(symbol)
        return "".join(output)
    if operation == "rotate_left":
        return symbols[1:] + symbols[:1]
    raise ValueError(f"unsupported symbolic operation: {operation}")


def make_symbolic_task(symbols: str, operation: SymbolicOperation) -> SymbolicTask:
    if not symbols:
        raise ValueError("symbols cannot be empty")
    if any(not symbol.isalpha() or not symbol.isascii() for symbol in symbols):
        raise ValueError("symbols must contain ASCII letters only")
    normalized = symbols.upper()
    answer = apply_symbolic_operation(normalized, operation)
    prompt = (
        f"Given the symbol sequence {normalized}, {_RULE_TEXT[operation]}. "
        "Return only the transformed symbol sequence with no spaces or explanation."
    )
    return SymbolicTask(
        prompt=prompt,
        answer=answer,
        operation=operation,
        length=len(normalized),
        symbols=normalized,
    )


def generate_symbolic_task(
    rng: Random,
    operation: SymbolicOperation,
    length: int,
    *,
    alphabet: Sequence[str] = ("A", "B", "C", "D", "E"),
) -> SymbolicTask:
    if length < 1:
        raise ValueError("length must be positive")
    normalized_alphabet = tuple(symbol.upper() for symbol in alphabet)
    if not normalized_alphabet:
        raise ValueError("alphabet cannot be empty")
    if any(len(symbol) != 1 or not symbol.isalpha() or not symbol.isascii() for symbol in normalized_alphabet):
        raise ValueError("alphabet entries must be single ASCII letters")
    symbols = "".join(rng.choice(normalized_alphabet) for _ in range(length))
    return make_symbolic_task(symbols, operation)


def symbolic_task_identity(task: SymbolicTask) -> tuple[str, str]:
    return task.operation, task.symbols
