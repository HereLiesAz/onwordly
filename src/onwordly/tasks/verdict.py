"""The verdict arithmetic language game (Experiment 009).

A verdict move shows a problem and a proposed answer and asks for a judgement:
``right`` if the proposal is correct, otherwise ``wrong: <correct integer>``.
Unlike Experiment 008's corrective move, the target depends on noticing
whether the proposal is right, so neither copying nor ignoring the proposal
can win once right and wrong proposals are balanced.

Verification-then-repair is established prior work (self-verification,
generative verifiers, critique-and-revise); 009 tests it under exact
verification and matched budgets, it does not claim the technique.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.verifiers.arithmetic import parse_integer_answer

_WRONG = re.compile(r"wrong:\s*(\S+)")


@dataclass(frozen=True, slots=True)
class VerdictTask:
    prompt: str
    answer: int
    shown: int
    base: ArithmeticTask

    @property
    def shown_is_right(self) -> bool:
        return self.shown == self.answer

    @property
    def target_text(self) -> str:
        return "right" if self.shown_is_right else f"wrong: {self.answer}"

    @property
    def bucket_key(self) -> str:
        return f"verdict-{'right' if self.shown_is_right else 'wrong'}:{self.base.bucket_key}"

    def verify(self, response: str) -> bool:
        return parse_verdict(response) == (True, None) if self.shown_is_right else (
            parse_verdict(response) == (False, self.answer)
        )


def parse_verdict(response: str) -> tuple[bool, int | None] | None:
    """``(True, None)`` for ``right``; ``(False, n)`` for ``wrong: n``; else ``None``."""
    text = response.strip()
    if text == "right":
        return True, None
    match = _WRONG.fullmatch(text)
    if match is None:
        return None
    value = parse_integer_answer(match.group(1))
    return None if value is None else (False, value)


def make_verdict_task(base: ArithmeticTask, shown: int) -> VerdictTask:
    prompt = (
        f"{base.prompt.rstrip()} A proposed answer is {shown}. "
        "If it is right, reply exactly: right. "
        "If it is wrong, reply exactly: wrong: <the correct integer>."
    )
    return VerdictTask(prompt=prompt, answer=base.answer, shown=shown, base=base)


def verify_task(task: object, response: str) -> bool:
    """Exact verification for any arithmetic-family task."""
    verify = getattr(task, "verify", None)
    if callable(verify):
        return bool(verify(response))
    parsed = parse_integer_answer(response)
    return parsed is not None and parsed == getattr(task, "answer")
