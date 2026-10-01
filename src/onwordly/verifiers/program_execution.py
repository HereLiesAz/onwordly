from __future__ import annotations

from onwordly.tasks.program_execution import ProgramTask
from onwordly.verifiers.arithmetic import parse_integer_answer


def verify_program_answer(task: ProgramTask, response: str) -> bool:
    parsed = parse_integer_answer(response)
    return parsed is not None and parsed == task.answer
