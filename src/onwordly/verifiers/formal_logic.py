from __future__ import annotations

from onwordly.tasks.formal_logic import LogicTask


def parse_logic_answer(response: str) -> str | None:
    """Accept only true/false; case is normalised by design."""
    stripped = response.strip().lower()
    return stripped if stripped in {"true", "false"} else None


def verify_logic_answer(task: LogicTask, response: str) -> bool:
    return parse_logic_answer(response) == task.answer
