from __future__ import annotations

import re

from onwordly.tasks.arithmetic import ArithmeticTask

_INTEGER = re.compile(r"^[+-]?\d+$")


def parse_integer_answer(response: str) -> int | None:
    """Parse a deliberately strict integer-only response."""
    value = response.strip()
    if not _INTEGER.fullmatch(value):
        return None
    return int(value)


def verify_arithmetic_answer(task: ArithmeticTask, response: str) -> bool:
    parsed = parse_integer_answer(response)
    return parsed == task.answer
