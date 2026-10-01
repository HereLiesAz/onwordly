from __future__ import annotations

from collections import deque
from random import Random
from typing import Protocol, Sequence

from onwordly.curricula.string_manipulation import AdaptiveStringCurriculum
from onwordly.tasks.string_manipulation import StringTask, make_string_task


class StringCurriculum(Protocol):
    alphabet: Sequence[str]

    def generate(self, rng: Random) -> StringTask:
        ...

    def update(self, task: StringTask, correct: bool) -> None:
        ...

    def accepts(self, task: StringTask) -> bool:
        ...


class StaticStringSource:
    def __init__(self, tasks: Sequence[StringTask]) -> None:
        if not tasks:
            raise ValueError("static dataset cannot be empty")
        self.tasks = tuple(tasks)
        self.index = 0

    def next_task(self, rng: Random) -> StringTask:
        del rng
        task = self.tasks[self.index % len(self.tasks)]
        self.index += 1
        return task

    def observe(self, task: StringTask, correct: bool) -> None:
        del task, correct


class OnlineStringSource:
    def __init__(self, curriculum: StringCurriculum | None = None) -> None:
        self.curriculum = curriculum or AdaptiveStringCurriculum()

    def next_task(self, rng: Random) -> StringTask:
        return self.curriculum.generate(rng)

    def observe(self, task: StringTask, correct: bool) -> None:
        self.curriculum.update(task, correct)


class ErrorFocusedStringSource(OnlineStringSource):
    def __init__(
        self,
        curriculum: StringCurriculum | None = None,
        *,
        variants_per_failure: int = 4,
    ) -> None:
        super().__init__(curriculum)
        if variants_per_failure < 1:
            raise ValueError("variants_per_failure must be positive")
        self.variants_per_failure = variants_per_failure
        self.pending: deque[StringTask] = deque()
        self._last_was_pending = False

    def next_task(self, rng: Random) -> StringTask:
        if self.pending:
            self._last_was_pending = True
            return self.pending.popleft()
        self._last_was_pending = False
        return super().next_task(rng)

    def observe(self, task: StringTask, correct: bool) -> None:
        super().observe(task, correct)
        if correct or self._last_was_pending:
            return
        self.pending.extend(self._nearby_variants(task)[: self.variants_per_failure])

    def _nearby_variants(self, task: StringTask) -> list[StringTask]:
        variants: list[StringTask] = []
        seen = {task.text}
        alphabet = tuple(char.upper() for char in self.curriculum.alphabet)
        for index, current in enumerate(task.text):
            for replacement in alphabet:
                if replacement == current:
                    continue
                text = task.text[:index] + replacement + task.text[index + 1 :]
                if text in seen:
                    continue
                seen.add(text)
                candidate = make_string_task(text, task.operation)
                if self.curriculum.accepts(candidate):
                    variants.append(candidate)
        return variants
