"""The corrective arithmetic language game.

A corrective task shows an arithmetic problem together with a previous answer
and asks for the correct integer: fix it if it is wrong, repeat it if it is
right. The target is always the true answer, so the existing exact integer
verifier applies unchanged.

Self-correction training is established prior work (e.g. self-refinement and
learning-from-mistakes methods); Experiment 008 measures it under exact
verification and matched token budgets, it does not claim the technique.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Literal

from onwordly.tasks.arithmetic import ArithmeticTask

CorrectiveKind = Literal["correct", "confirm"]


@dataclass(frozen=True, slots=True)
class CorrectiveTask:
    prompt: str
    answer: int
    previous: int
    kind: CorrectiveKind
    base: ArithmeticTask

    @property
    def target_text(self) -> str:
        return str(self.answer)

    @property
    def bucket_key(self) -> str:
        return f"{self.kind}:{self.base.bucket_key}"


def make_corrective_task(base: ArithmeticTask, previous: int) -> CorrectiveTask:
    kind: CorrectiveKind = "confirm" if previous == base.answer else "correct"
    prompt = (
        f"{base.prompt.rstrip()} A previous answer was {previous}. "
        "If it is wrong, return the correct integer; if it is right, return it unchanged."
    )
    return CorrectiveTask(prompt=prompt, answer=base.answer, previous=previous, kind=kind, base=base)


def synthetic_wrong_answer(base: ArithmeticTask, rng: Random) -> int:
    """A plausible wrong answer: off by one, off by ten, a carry slip or a digit swap."""
    answer = base.answer
    candidates = {answer + 1, answer - 1, answer + 10, answer - 10}
    digits = str(abs(answer))
    if len(digits) >= 2:
        swapped = digits[:-2] + digits[-1] + digits[-2]
        candidates.add(int(swapped) * (-1 if answer < 0 else 1))
        candidates.add(answer + 100 if len(digits) >= 3 else answer - 9)
    candidates.discard(answer)
    return rng.choice(sorted(candidates))
