from __future__ import annotations

from collections import deque
from random import Random
from typing import Protocol, Sequence

from onwordly.curricula.program_execution import AdaptiveProgramCurriculum
from onwordly.tasks.program_execution import Instruction, ProgramTask, make_program_task


class ProgramCurriculum(Protocol):
    argument_min: int
    argument_max: int

    def generate(self, rng: Random) -> ProgramTask:
        ...

    def update(self, task: ProgramTask, correct: bool) -> None:
        ...

    def accepts(self, task: ProgramTask) -> bool:
        ...


class StaticProgramSource:
    def __init__(self, tasks: Sequence[ProgramTask]) -> None:
        if not tasks:
            raise ValueError("static dataset cannot be empty")
        self.tasks = tuple(tasks)
        self.index = 0

    def next_task(self, rng: Random) -> ProgramTask:
        del rng
        task = self.tasks[self.index % len(self.tasks)]
        self.index += 1
        return task

    def observe(self, task: ProgramTask, correct: bool) -> None:
        del task, correct


class OnlineProgramSource:
    def __init__(self, curriculum: ProgramCurriculum | None = None) -> None:
        self.curriculum = curriculum or AdaptiveProgramCurriculum()

    def next_task(self, rng: Random) -> ProgramTask:
        return self.curriculum.generate(rng)

    def observe(self, task: ProgramTask, correct: bool) -> None:
        self.curriculum.update(task, correct)


class ErrorFocusedProgramSource(OnlineProgramSource):
    def __init__(
        self,
        curriculum: ProgramCurriculum | None = None,
        *,
        variants_per_failure: int = 4,
    ) -> None:
        super().__init__(curriculum)
        if variants_per_failure < 1:
            raise ValueError("variants_per_failure must be positive")
        self.variants_per_failure = variants_per_failure
        self.pending: deque[ProgramTask] = deque()
        self._last_was_pending = False

    def next_task(self, rng: Random) -> ProgramTask:
        if self.pending:
            self._last_was_pending = True
            return self.pending.popleft()
        self._last_was_pending = False
        return super().next_task(rng)

    def observe(self, task: ProgramTask, correct: bool) -> None:
        super().observe(task, correct)
        if correct or self._last_was_pending:
            return
        self.pending.extend(self._nearby_variants(task)[: self.variants_per_failure])

    def _nearby_variants(self, task: ProgramTask) -> list[ProgramTask]:
        variants: list[ProgramTask] = []
        seen = {task.program_text}
        for index, instruction in enumerate(task.instructions):
            if instruction.argument is None:
                continue
            for delta in (-1, 1):
                argument = instruction.argument + delta
                if not (self.curriculum.argument_min <= argument <= self.curriculum.argument_max):
                    continue
                instructions = list(task.instructions)
                instructions[index] = Instruction(instruction.name, argument)
                candidate = make_program_task(instructions)
                if candidate.program_text in seen:
                    continue
                seen.add(candidate.program_text)
                if self.curriculum.accepts(candidate):
                    variants.append(candidate)
        return variants
