from __future__ import annotations

from collections import deque
from random import Random
from typing import Protocol, Sequence

from onwordly.curricula.symbolic import AdaptiveSymbolicCurriculum
from onwordly.tasks.symbolic import SymbolicTask, make_symbolic_task


class SymbolicCurriculum(Protocol):
    alphabet: Sequence[str]

    def generate(self, rng: Random) -> SymbolicTask:
        ...

    def update(self, task: SymbolicTask, correct: bool) -> None:
        ...

    def accepts(self, task: SymbolicTask) -> bool:
        ...


class StaticSymbolicSource:
    def __init__(self, tasks: Sequence[SymbolicTask]) -> None:
        if not tasks:
            raise ValueError("static dataset cannot be empty")
        self.tasks = tuple(tasks)
        self.index = 0

    def next_task(self, rng: Random) -> SymbolicTask:
        del rng
        task = self.tasks[self.index % len(self.tasks)]
        self.index += 1
        return task

    def observe(self, task: SymbolicTask, correct: bool) -> None:
        del task, correct


class OnlineSymbolicSource:
    def __init__(self, curriculum: SymbolicCurriculum | None = None) -> None:
        self.curriculum = curriculum or AdaptiveSymbolicCurriculum()

    def next_task(self, rng: Random) -> SymbolicTask:
        return self.curriculum.generate(rng)

    def observe(self, task: SymbolicTask, correct: bool) -> None:
        self.curriculum.update(task, correct)


class ErrorFocusedSymbolicSource(OnlineSymbolicSource):
    def __init__(
        self,
        curriculum: SymbolicCurriculum | None = None,
        *,
        variants_per_failure: int = 4,
    ) -> None:
        super().__init__(curriculum)
        if variants_per_failure < 1:
            raise ValueError("variants_per_failure must be positive")
        self.variants_per_failure = variants_per_failure
        self.pending: deque[SymbolicTask] = deque()
        self._last_was_pending = False

    def next_task(self, rng: Random) -> SymbolicTask:
        if self.pending:
            self._last_was_pending = True
            return self.pending.popleft()
        self._last_was_pending = False
        return super().next_task(rng)

    def observe(self, task: SymbolicTask, correct: bool) -> None:
        super().observe(task, correct)
        if correct or self._last_was_pending:
            return
        self.pending.extend(self._nearby_variants(task)[: self.variants_per_failure])

    def _nearby_variants(self, task: SymbolicTask) -> list[SymbolicTask]:
        variants: list[SymbolicTask] = []
        seen = {task.symbols}
        alphabet = tuple(symbol.upper() for symbol in self.curriculum.alphabet)
        for index, current in enumerate(task.symbols):
            for replacement in alphabet:
                if replacement == current:
                    continue
                symbols = task.symbols[:index] + replacement + task.symbols[index + 1 :]
                if symbols in seen:
                    continue
                seen.add(symbols)
                candidate = make_symbolic_task(symbols, task.operation)
                if self.curriculum.accepts(candidate):
                    variants.append(candidate)
        return variants
