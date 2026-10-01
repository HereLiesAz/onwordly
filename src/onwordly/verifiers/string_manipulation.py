from __future__ import annotations

from typing import Protocol


class StringAnswerTask(Protocol):
    answer: str


def verify_string_answer(task: StringAnswerTask, response: str) -> bool:
    return response.strip() == task.answer
