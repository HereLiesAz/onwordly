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
    parsed = tuple(parse_integer_answer(part) for part in parts)
    if any(value is None for value in parsed):
        return None
    return parsed  # type: ignore[return-value]


def verify_program_trace(task: ProgramTraceTask, response: str) -> bool:
    return parse_program_trace(response) == task.states


def parse_supervised_program_final(response: str) -> int | None:
    """Parse ``FINAL=<integer>`` or a well-formed ``TRACE=<states>;FINAL=<integer>``.

    Arbitrary text before ``FINAL=`` is rejected; a trace prefix must parse, though
    its correctness is judged only by ``verify_supervised_program_trace``.
    """
    stripped = response.strip()
    if stripped.startswith("TRACE="):
        if ";FINAL=" not in stripped:
            return None
        trace_text, stripped = stripped[len("TRACE="):].split(";FINAL=", 1)
        if parse_program_trace(trace_text) is None:
            return None
        stripped = "FINAL=" + stripped
    marker = "FINAL="
    if not stripped.startswith(marker):
        return None
    suffix = stripped[len(marker):]
    if suffix.strip() != suffix:
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
