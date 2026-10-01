from __future__ import annotations

from collections import deque
from random import Random
from typing import Protocol, Sequence

from onwordly.curricula.adaptive import AdaptiveArithmeticCurriculum
from onwordly.tasks.arithmetic import ArithmeticTask, make_arithmetic_task, operand_bounds


class ArithmeticTaskSource(Protocol):
    def next_task(self, rng: Random) -> ArithmeticTask:
        ...

    def observe(self, task: ArithmeticTask, correct: bool) -> None:
        ...


class StaticArithmeticSource:
    def __init__(self, tasks: Sequence[ArithmeticTask]) -> None:
        if not tasks:
            raise ValueError("static dataset cannot be empty")
        self.tasks = tuple(tasks)
        self.index = 0

    def next_task(self, rng: Random) -> ArithmeticTask:
        del rng
        task = self.tasks[self.index % len(self.tasks)]
        self.index += 1
        return task

    def observe(self, task: ArithmeticTask, correct: bool) -> None:
        del task, correct


class AdaptiveArithmeticSource:
    def __init__(self, curriculum: AdaptiveArithmeticCurriculum | None = None) -> None:
        self.curriculum = curriculum or AdaptiveArithmeticCurriculum()

    def next_task(self, rng: Random) -> ArithmeticTask:
        return self.curriculum.generate(rng)

    def observe(self, task: ArithmeticTask, correct: bool) -> None:
        self.curriculum.update(task, correct)


class ErrorFocusedArithmeticSource(AdaptiveArithmeticSource):
    """Adaptive source that queues nearby variants after a failed attempt.

    This is intentionally an application of established hard-example/counterexample
    training ideas, not a novelty claim.
    """

    def __init__(
        self,
        curriculum: AdaptiveArithmeticCurriculum | None = None,
        *,
        variants_per_failure: int = 4,
    ) -> None:
        super().__init__(curriculum)
        if variants_per_failure < 1:
            raise ValueError("variants_per_failure must be at least 1")
        self.variants_per_failure = variants_per_failure
        self.pending: deque[ArithmeticTask] = deque()
        self._last_was_pending = False

    def next_task(self, rng: Random) -> ArithmeticTask:
        if self.pending:
            self._last_was_pending = True
            return self.pending.popleft()
        self._last_was_pending = False
        return super().next_task(rng)

    def observe(self, task: ArithmeticTask, correct: bool) -> None:
        super().observe(task, correct)
        if correct or self._last_was_pending:
            return
        self.pending.extend(self._nearby_variants(task)[: self.variants_per_failure])

    def _nearby_variants(self, task: ArithmeticTask) -> list[ArithmeticTask]:
        low, high = operand_bounds(task.digits)
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))
        variants: list[ArithmeticTask] = []
        seen: set[tuple[int, int]] = {(task.left, task.right)}

        for left_delta, right_delta in offsets:
            left = task.left + left_delta
            right = task.right + right_delta
            if not (low <= left <= high and low <= right <= high):
                continue
            candidate = make_arithmetic_task(
                left,
                right,
                task.operation,
                prompt_style=task.prompt_style,
            )
            key = (candidate.left, candidate.right)
            if key in seen:
                continue
            seen.add(key)
            if candidate.digits == task.digits and self.curriculum.accepts(candidate):
                variants.append(candidate)
        return variants
