from __future__ import annotations

from onwordly.tasks.program_execution import ProgramTask, ProgramTraceTask
from onwordly.verifiers.arithmetic import parse_integer_answer


def verify_program_answer(task: ProgramTask, response: str) -> bool:
    parsed = parse_integer_answer(response)
    return parsed is not None and parsed == task.answer


def parse_program_trace(response: str) -> tuple[int, ...] | None:
    stripped = response.strip()
    if not stripped:
        return None
    parts = stripped.split(",")
    if any(part.strip() != part or not part for part in parts):
        return None
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return None


def verify_program_trace(task: ProgramTraceTask, response: str) -> bool:
    return parse_program_trace(response) == task.states
