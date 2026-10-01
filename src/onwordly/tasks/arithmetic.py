from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from random import Random
from typing import Literal

Operation = Literal["add", "subtract", "multiply"]
PromptStyle = Literal["canonical", "question", "words", "expression"]
ArithmeticPartition = Literal["train", "eval"]

PROMPT_STYLES: tuple[PromptStyle, ...] = (
    "canonical",
    "question",
    "words",
    "expression",
)


@dataclass(frozen=True, slots=True)
class ArithmeticTask:
    prompt: str
    answer: int
    operation: Operation
    digits: int
    left: int
    right: int
    prompt_style: PromptStyle = "canonical"

    @property
    def target_text(self) -> str:
        return str(self.answer)

    @property
    def bucket_key(self) -> str:
        return f"{self.operation}:{self.digits}"


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


def _partition_operands(left: int, right: int, operation: Operation) -> tuple[int, int]:
    if operation in ("add", "multiply") and right < left:
        return right, left
    if operation == "subtract" and right > left:
        return right, left
    return left, right


def arithmetic_partition(
    left: int,
    right: int,
    operation: Operation,
    *,
    modulus: int = 5,
) -> ArithmeticPartition:
    """Assign an operand combination to a stable train/eval partition.

    Commutative operations normalize operand order, so 2+3 and 3+2 cannot leak
    across the split.
    """
    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    partition_left, partition_right = _partition_operands(left, right, operation)
    key = f"{operation}:{partition_left}:{partition_right}".encode("utf-8")
    residue = int.from_bytes(blake2b(key, digest_size=8).digest(), "big") % modulus
    return "eval" if residue == 0 else "train"


def task_in_partition(
    task: ArithmeticTask,
    partition: ArithmeticPartition | None,
    *,
    modulus: int = 5,
) -> bool:
    if partition is None:
        return True
    return arithmetic_partition(
        task.left,
        task.right,
        task.operation,
        modulus=modulus,
    ) == partition


def arithmetic_task_identity(
    task: ArithmeticTask,
    *,
    include_prompt_style: bool = False,
) -> tuple[str, int, int] | tuple[str, int, int, str]:
    """Return a stable logical identity for duplicate/leakage diagnostics.

    Commutative operations share one operand identity regardless of presentation
    order. Prompt style can optionally be included when the presented example,
    rather than the underlying arithmetic problem, is the unit of analysis.
    """
    left, right = _partition_operands(task.left, task.right, task.operation)
    base: tuple[str, int, int] = (task.operation, left, right)
    if include_prompt_style:
        return (*base, task.prompt_style)
    return base


def _format_prompt(
    left: int,
    right: int,
    operation: Operation,
    prompt_style: PromptStyle,
) -> str:
    symbol = _SYMBOLS[operation]
    if prompt_style == "canonical":
        return f"Compute {left} {symbol} {right}. Return only the integer answer."
    if prompt_style == "question":
        return f"What is {left} {symbol} {right}? Answer with only the integer."
    if prompt_style == "expression":
        return f"{left} {symbol} {right} = ?\nInteger only."
    if prompt_style == "words":
        if operation == "add":
            return f"Add {left} and {right}. Return only the integer result."
        if operation == "subtract":
            return f"Subtract {right} from {left}. Return only the integer result."
        if operation == "multiply":
            return f"Multiply {left} by {right}. Return only the integer result."
    raise ValueError(f"unsupported prompt style: {prompt_style}")


def make_arithmetic_task(
    left: int,
    right: int,
    operation: Operation,
    *,
    prompt_style: PromptStyle = "canonical",
) -> ArithmeticTask:
    if left < 0 or right < 0:
        raise ValueError("operands must be non-negative")
    if operation == "subtract" and right > left:
        left, right = right, left

    if operation == "add":
        answer = left + right
    elif operation == "subtract":
        answer = left - right
    elif operation == "multiply":
        answer = left * right
    else:
        raise ValueError(f"unsupported operation: {operation}")

    digits = max(len(str(left)), len(str(right)))
    prompt = _format_prompt(left, right, operation, prompt_style)

    return ArithmeticTask(
        prompt=prompt,
        answer=answer,
        operation=operation,
        digits=digits,
        left=left,
        right=right,
        prompt_style=prompt_style,
    )


def generate_arithmetic_task(
    rng: Random,
    operation: Operation,
    digits: int,
    *,
    prompt_style: PromptStyle = "canonical",
) -> ArithmeticTask:
    """Generate one exactly verifiable integer-arithmetic task."""
    low, high = operand_bounds(digits)
    left = rng.randint(low, high)
    right = rng.randint(low, high)
    return make_arithmetic_task(
        left,
        right,
        operation,
        prompt_style=prompt_style,
    )
