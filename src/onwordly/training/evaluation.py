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
    by_prompt_style: dict[str, dict[str, float | int]]

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
    styles: dict[str, list[int]] = {}

    for task in tasks:
        response = adapter.generate(task.prompt)
        is_correct = verify_arithmetic_answer(task, response)
        correct += int(is_correct)

        bucket_key = f"{task.operation}:{task.digits}"
        bucket = buckets.setdefault(bucket_key, [0, 0])
        bucket[0] += 1
        bucket[1] += int(is_correct)

        style = styles.setdefault(task.prompt_style, [0, 0])
        style[0] += 1
        style[1] += int(is_correct)

    def summarize(records: dict[str, list[int]]) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "examples": values[0],
                "correct": values[1],
                "accuracy": values[1] / values[0],
            }
            for key, values in sorted(records.items())
        }

    return EvaluationResult(
        examples=len(tasks),
        correct=correct,
        accuracy=correct / len(tasks),
        by_bucket=summarize(buckets),
        by_prompt_style=summarize(styles),
    )
