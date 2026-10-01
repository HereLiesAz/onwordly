from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from onwordly.models.base import ModelAdapter
from onwordly.tasks.arithmetic import ArithmeticTask
from onwordly.verifiers.arithmetic import verify_arithmetic_answer


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    examples: int
    correct: int
    accuracy: float
    by_bucket: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_arithmetic(
    adapter: ModelAdapter,
    tasks: Sequence[ArithmeticTask],
) -> EvaluationResult:
    if not tasks:
        raise ValueError("evaluation set cannot be empty")

    correct = 0
    buckets: dict[str, list[int]] = {}
    for task in tasks:
        response = adapter.generate(task.prompt)
        is_correct = verify_arithmetic_answer(task, response)
        correct += int(is_correct)
        key = f"{task.operation}:{task.digits}"
        record = buckets.setdefault(key, [0, 0])
        record[0] += 1
        record[1] += int(is_correct)

    by_bucket = {
        key: {
            "examples": values[0],
            "correct": values[1],
            "accuracy": values[1] / values[0],
        }
        for key, values in sorted(buckets.items())
    }
    return EvaluationResult(
        examples=len(tasks),
        correct=correct,
        accuracy=correct / len(tasks),
        by_bucket=by_bucket,
    )
