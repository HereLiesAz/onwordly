from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Literal

Operation = Literal["add", "subtract", "multiply"]


@dataclass(frozen=True, slots=True)
class ArithmeticTask:
    prompt: str
    answer: int
    operation: Operation
    digits: int
    left: int
    right: int


_SYMBOLS: dict[Operation, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
}


def _operand_bounds(digits: int) -> tuple[int, int]:
    if digits < 1:
        raise ValueError("digits must be at least 1")
    if digits == 1:
        return 0, 9
    return 10 ** (digits - 1), (10**digits) - 1


def generate_arithmetic_task(
    rng: Random,
    operation: Operation,
    digits: int,
) -> ArithmeticTask:
    """Generate one exactly verifiable integer-arithmetic task."""
    low, high = _operand_bounds(digits)
    left = rng.randint(low, high)
    right = rng.randint(low, high)

    if operation == "add":
        answer = left + right
    elif operation == "subtract":
        if right > left:
            left, right = right, left
        answer = left - right
    elif operation == "multiply":
        answer = left * right
    else:
        raise ValueError(f"unsupported operation: {operation}")

    symbol = _SYMBOLS[operation]
    prompt = f"Compute {left} {symbol} {right}. Return only the integer answer."

    return ArithmeticTask(
        prompt=prompt,
        answer=answer,
        operation=operation,
        digits=digits,
        left=left,
        right=right,
    )
