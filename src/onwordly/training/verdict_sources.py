"""Training source for Experiment 009 (verdict arithmetic language game)."""
from __future__ import annotations

from collections import deque
from random import Random
from typing import Literal, Sequence

from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.tasks.corrective import synthetic_wrong_answer
from onwordly.tasks.verdict import VerdictTask, make_verdict_task
from onwordly.training.sources import StaticArithmeticSource
from onwordly.verifiers.arithmetic import parse_integer_answer

WrongSource = Literal["synthetic", "mixed"]


class VerdictArithmeticSource:
    """Static arithmetic stream with balanced verdict moves interleaved.

    After the pre-update attempt on each arithmetic task, with probability
    ``verdict_rate`` a verdict move on that task is queued. Its proposal is the
    true answer with probability 0.5, otherwise a wrong answer:
    - ``synthetic``: always a synthetic near miss;
    - ``mixed``: the model's own wrong answer when its attempt was wrong and
      parseable, otherwise a synthetic near miss.
    The trigger does not depend on whether the attempt was right, so the
    right/wrong balance holds by construction. Built only from training tasks.
    """

    def __init__(
        self,
        tasks: Sequence[ArithmeticTask],
        *,
        wrong_source: WrongSource,
        verdict_rate: float,
        seed: int,
    ) -> None:
        if wrong_source not in ("synthetic", "mixed"):
            raise ValueError("wrong_source must be 'synthetic' or 'mixed'")
        if not 0.0 < verdict_rate <= 1.0:
            raise ValueError("verdict_rate must be in (0, 1]")
        self.base = StaticArithmeticSource(tasks)
        self.wrong_source = wrong_source
        self.verdict_rate = verdict_rate
        self.rng = Random(seed)
        self.pending: deque[VerdictTask] = deque()
        self.queued = {"right": 0, "wrong_synthetic": 0, "wrong_own": 0}

    def next_task(self, rng: Random) -> ArithmeticTask | VerdictTask:
        if self.pending:
            return self.pending.popleft()
        return self.base.next_task(rng)

    def observe(self, task: object, correct: bool) -> None:
        del task, correct

    def observe_response(self, task: object, response: str, correct: bool) -> None:
        if not isinstance(task, ArithmeticTask) or self.rng.random() >= self.verdict_rate:
            return
        if self.rng.random() < 0.5:
            self.pending.append(make_verdict_task(task, task.answer))
            self.queued["right"] += 1
            return
        parsed = parse_integer_answer(response)
        if self.wrong_source == "mixed" and not correct and parsed is not None and parsed != task.answer:
            self.pending.append(make_verdict_task(task, parsed))
            self.queued["wrong_own"] += 1
            return
        self.pending.append(make_verdict_task(task, synthetic_wrong_answer(task, self.rng)))
        self.queued["wrong_synthetic"] += 1
