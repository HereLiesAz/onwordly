"""Solve-then-judge arithmetic move (Experiment 010, ordering lever).

Same information as Experiment 009's verdict move, but the reply puts the
model's own answer *before* the judgement: ``<correct integer>; right`` when
the shown proposal is correct, ``<correct integer>; wrong`` when it is not.
A model that writes its answer first can judge by comparing, rather than
having to decide before computing.

Answer-before-verdict ordering is an established idea (generative verifiers
and chain-of-thought verification); 010 tests it under exact verification and
a matched token budget, it does not claim the technique.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.verifiers.arithmetic import parse_integer_answer

_REPLY = re.compile(r"(\S+);\s*(right|wrong)")


@dataclass(frozen=True, slots=True)
class SolveJudgeTask:
    prompt: str
    answer: int
    shown: int
    base: ArithmeticTask

    @property
    def shown_is_right(self) -> bool:
        return self.shown == self.answer

    @property
    def target_text(self) -> str:
        return f"{self.answer}; {'right' if self.shown_is_right else 'wrong'}"

    @property
    def bucket_key(self) -> str:
        return f"solve-judge-{'right' if self.shown_is_right else 'wrong'}:{self.base.bucket_key}"

    def verify(self, response: str) -> bool:
        return parse_solve_judge(response) == (self.answer, self.shown_is_right)


def parse_solve_judge(response: str) -> tuple[int, bool] | None:
    """``(n, True)`` for ``n; right``; ``(n, False)`` for ``n; wrong``; else ``None``.

    ``n`` must be a canonical integer (same rule as the arithmetic verifier).
    """
    match = _REPLY.fullmatch(response.strip())
    if match is None:
        return None
    value = parse_integer_answer(match.group(1))
    return None if value is None else (value, match.group(2) == "right")


def make_solve_judge_task(base: ArithmeticTask, shown: int) -> SolveJudgeTask:
    prompt = (
        f"{base.prompt.rstrip()} A proposed answer is {shown}. "
        "First give the correct integer, then judge the proposal. "
        "Reply exactly: <the correct integer>; right "
        "or exactly: <the correct integer>; wrong."
    )
    return SolveJudgeTask(prompt=prompt, answer=base.answer, shown=shown, base=base)
