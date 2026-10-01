from __future__ import annotations

from collections import deque
from random import Random
from typing import Protocol, Sequence

from onwordly.curricula.formal_logic import AdaptiveLogicCurriculum
from onwordly.tasks.formal_logic import LogicTask, make_logic_task


class LogicCurriculum(Protocol):
    def generate(self, rng: Random) -> LogicTask:
        ...

    def update(self, task: LogicTask, correct: bool) -> None:
        ...

    def accepts(self, task: LogicTask) -> bool:
        ...


class StaticLogicSource:
    def __init__(self, tasks: Sequence[LogicTask]) -> None:
        if not tasks:
            raise ValueError("static dataset cannot be empty")
        self.tasks = tuple(tasks)
        self.index = 0

    def next_task(self, rng: Random) -> LogicTask:
        del rng
        task = self.tasks[self.index % len(self.tasks)]
        self.index += 1
        return task

    def observe(self, task: LogicTask, correct: bool) -> None:
        del task, correct


class OnlineLogicSource:
    def __init__(self, curriculum: LogicCurriculum | None = None) -> None:
        self.curriculum = curriculum or AdaptiveLogicCurriculum()

    def next_task(self, rng: Random) -> LogicTask:
        return self.curriculum.generate(rng)

    def observe(self, task: LogicTask, correct: bool) -> None:
        self.curriculum.update(task, correct)


class ErrorFocusedLogicSource(OnlineLogicSource):
    def __init__(
        self,
        curriculum: LogicCurriculum | None = None,
        *,
        variants_per_failure: int = 4,
    ) -> None:
        super().__init__(curriculum)
        if variants_per_failure < 1:
            raise ValueError("variants_per_failure must be positive")
        self.variants_per_failure = variants_per_failure
        self.pending: deque[LogicTask] = deque()
        self._last_was_pending = False

    def next_task(self, rng: Random) -> LogicTask:
        if self.pending:
            self._last_was_pending = True
            return self.pending.popleft()
        self._last_was_pending = False
        return super().next_task(rng)

    def observe(self, task: LogicTask, correct: bool) -> None:
        super().observe(task, correct)
        if correct or self._last_was_pending:
            return
        self.pending.extend(self._nearby_variants(task)[: self.variants_per_failure])

    def _nearby_variants(self, task: LogicTask) -> list[LogicTask]:
        variants: list[LogicTask] = []
        assignment = dict(task.assignment)
        for name, value in task.assignment:
            mutated = dict(assignment)
            mutated[name] = not value
            candidate = make_logic_task(task.expression, mutated)
            if self.curriculum.accepts(candidate):
                variants.append(candidate)
        return variants
