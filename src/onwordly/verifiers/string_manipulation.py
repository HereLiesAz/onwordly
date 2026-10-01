from __future__ import annotations

from onwordly.tasks.string_manipulation import StringTask


def verify_string_answer(task: StringTask, response: str) -> bool:
    return response.strip() == task.answer
