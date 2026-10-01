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


def operand_bounds(digits: int) -> tuple[int, int]:
    if digits < 1:
        raise ValueError("digits must be at least 1")
    if digits == 1:
        return 0, 9
    return 10 ** (digits - 1), (10**digits) - 1


def make_arithmetic_task(left: int, right: int, operation: Operation) -> ArithmeticTask:
    if left < 0 or right < 0:
        raise ValueError("operands must be non-negative")

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

    digits = max(len(str(left)), len(str(right)))
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


def generate_arithmetic_task(
    rng: Random,
    operation: Operation,
    digits: int,
) -> ArithmeticTask:
    """Generate one exactly verifiable integer-arithmetic task."""
    low, high = operand_bounds(digits)
    left = rng.randint(low, high)
    right = rng.randint(low, high)
    return make_arithmetic_task(left, right, operation)
