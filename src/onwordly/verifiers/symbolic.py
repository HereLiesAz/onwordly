from __future__ import annotations

from typing import Protocol


class SymbolicAnswerTask(Protocol):
    answer: str


def parse_symbolic_answer(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if any(not symbol.isalpha() or not symbol.isascii() for symbol in stripped):
        return None
    return stripped.upper()


def verify_symbolic_answer(task: SymbolicAnswerTask, response: str) -> bool:
    parsed = parse_symbolic_answer(response)
    return parsed == task.answer
