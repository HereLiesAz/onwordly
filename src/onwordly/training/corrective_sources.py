"""Training sources for Experiment 008 (corrective arithmetic language game)."""
from __future__ import annotations

from collections import deque
from random import Random
from typing import Literal, Sequence

from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.tasks.corrective import CorrectiveTask, make_corrective_task, synthetic_wrong_answer
from onwordly.training.sources import StaticArithmeticSource
from onwordly.verifiers.arithmetic import parse_integer_answer

PreviousAnswer = Literal["own", "synthetic"]


class CorrectiveArithmeticSource:
    """Static arithmetic stream with corrective moves interleaved.

    After the pre-update attempt on an arithmetic task:
    - wrong with a parseable integer -> queue a corrective task whose previous
      answer is the model's own (``own``) or a synthetic plausible error
      (``synthetic``); the trigger is identical in both arms, only the shown
      answer differs;
    - right -> with ``confirm_probability`` queue a confirm task showing the
      correct answer, so "previous answers are always wrong" is not learnable.
    Corrective tasks are built only from training-partition tasks and are not
    themselves followed up.
    """

    def __init__(
        self,
        tasks: Sequence[ArithmeticTask],
        *,
        previous: PreviousAnswer,
        confirm_probability: float,
        seed: int,
    ) -> None:
        if previous not in ("own", "synthetic"):
            raise ValueError("previous must be 'own' or 'synthetic'")
        if not 0.0 <= confirm_probability <= 1.0:
            raise ValueError("confirm_probability must be in [0, 1]")
        self.base = StaticArithmeticSource(tasks)
        self.previous = previous
        self.confirm_probability = confirm_probability
        self.rng = Random(seed)
        self.pending: deque[CorrectiveTask] = deque()
        self.queued = {"correct": 0, "confirm": 0}

    def next_task(self, rng: Random) -> ArithmeticTask | CorrectiveTask:
        if self.pending:
            return self.pending.popleft()
        return self.base.next_task(rng)

    def observe(self, task: object, correct: bool) -> None:
        del task, correct

    def observe_response(self, task: object, response: str, correct: bool) -> None:
        if not isinstance(task, ArithmeticTask):
            return
        if correct:
            if self.rng.random() < self.confirm_probability:
                self.pending.append(make_corrective_task(task, task.answer))
                self.queued["confirm"] += 1
            return
        parsed = parse_integer_answer(response)
        if parsed is None:
            return
        shown = parsed if self.previous == "own" else synthetic_wrong_answer(task, self.rng)
        self.pending.append(make_corrective_task(task, shown))
        self.queued["correct"] += 1
