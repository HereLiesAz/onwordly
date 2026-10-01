from __future__ import annotations

from onwordly.tasks.program_execution import (
    ProgramSupervisionTask,
    ProgramTask,
    ProgramTraceTask,
)
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


def parse_supervised_program_final(response: str) -> int | None:
    stripped = response.strip()
    marker = "FINAL="
    index = stripped.rfind(marker)
    if index < 0:
        return None
    suffix = stripped[index + len(marker):]
    if not suffix or ";" in suffix or "," in suffix or " " in suffix:
        return None
    return parse_integer_answer(suffix)


def verify_supervised_program_final(
    task: ProgramSupervisionTask,
    response: str,
) -> bool:
    return parse_supervised_program_final(response) == task.final_answer


def verify_supervised_program_trace(
    task: ProgramSupervisionTask,
    response: str,
) -> bool:
    stripped = response.strip()
    prefix = "TRACE="
    marker = ";FINAL="
    if not stripped.startswith(prefix) or marker not in stripped:
        return False
    trace_text, final_text = stripped[len(prefix):].split(marker, 1)
    trace = parse_program_trace(trace_text)
    final = parse_integer_answer(final_text)
    return trace == task.states and final == task.final_answer
